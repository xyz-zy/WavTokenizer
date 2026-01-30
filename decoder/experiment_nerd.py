import math

import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio
import transformers
import yaml

from decoder.discriminator_dac import DACDiscriminator

from decoder.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from decoder.feature_extractors import FeatureExtractor
from decoder.heads import FourierHead
from decoder.helpers import plot_spectrogram_to_numpy
from decoder.loss import DiscriminatorLoss, GeneratorLoss, FeatureMatchingLoss, MelSpecReconstructionLoss, DACGANLoss
from decoder.models import Backbone
from decoder.modules import safe_log
from decoder.pretrained_model import instantiate_class
from nerd.nerd import NERDConfig

from einops import rearrange

# from nerd.nerd import NERDConfig#, NERDSampler, NERDRDEstimator, RDEstimatorConfig

def plot_pca_components(X, codebook_vectors, random_seed, suffix: str = ""):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    # print(X.shape)

    n_comp = min(4, X.shape[1])
    pca = PCA(n_components=n_comp, random_state=random_seed)
    X_pca_full = pca.fit_transform(X)

    suf = f"_{suffix}" if suffix else ""

    def _plot_pca_pair(
        X2, cb2, filestem, codebook_name, x_label, y_label, title
    ):
        # np.save(out_folder / f"{filestem}.npy", X2)
        # if cb2 is not None:
        #     np.save(out_folder / f"{codebook_name}.npy", cb2)

        n_bins = 100

        fig = plt.figure(figsize=(12, 6), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, wspace=0.2)

        def _panel_axes(spec):
            sub = spec.subgridspec(
                2, 2, width_ratios=(4, 1), height_ratios=(1, 4), hspace=0.05, wspace=0.05
            )
            ax_histx = fig.add_subplot(sub[0, 0])
            ax_main = fig.add_subplot(sub[1, 0])
            ax_histy = fig.add_subplot(sub[1, 1])
            return ax_main, ax_histx, ax_histy

        def _plot_with_hists(ax_main, ax_histx, ax_histy, add_codebook, panel_title):
            ax_main.scatter(
                X2[:, 0], X2[:, 1], c="tab:blue", s=6, alpha=0.6, label="latents"
            )
            if add_codebook and cb2 is not None:
                ax_main.scatter(
                    cb2[:, 0],
                    cb2[:, 1],
                    c="tab:orange",
                    s=24,
                    alpha=0.95,
                    edgecolors="k",
                    label="codebook",
                )
            ax_main.set_xlabel(x_label)
            ax_main.set_ylabel(y_label)
            ax_main.set_title(panel_title)
            if add_codebook and cb2 is not None:
                ax_main.legend()

            ax_histx.hist(
                X2[:, 0], bins=n_bins, color="tab:blue", alpha=0.6, density=True
            )
            if add_codebook and cb2 is not None:
                ax_histx.hist(
                    cb2[:, 0], bins=n_bins, color="tab:orange", alpha=0.6, density=True
                )
            ax_histx.axis("off")

            ax_histy.hist(
                X2[:, 1],
                bins=n_bins,
                orientation="horizontal",
                color="tab:blue",
                alpha=0.6,
                density=True,
            )
            if add_codebook and cb2 is not None:
                ax_histy.hist(
                    cb2[:, 1],
                    bins=n_bins,
                    orientation="horizontal",
                    color="tab:orange",
                    alpha=0.6,
                    density=True,
                )
            ax_histy.axis("off")

        def _shared_limits():
            data = X2
            if cb2 is not None:
                data = np.concatenate([X2, cb2], axis=0)
            x_min = float(np.nanmin(data[:, 0]))
            x_max = float(np.nanmax(data[:, 0]))
            y_min = float(np.nanmin(data[:, 1]))
            y_max = float(np.nanmax(data[:, 1]))
            x_pad = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
            y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
            return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)

        xlim, ylim = _shared_limits()

        ax_main_l, ax_histx_l, ax_histy_l = _panel_axes(gs[0])
        _plot_with_hists(
            ax_main_l, ax_histx_l, ax_histy_l, add_codebook=False, panel_title="latents"
        )
        ax_main_l.set_xlim(xlim)
        ax_main_l.set_ylim(ylim)

        ax_main_r, ax_histx_r, ax_histy_r = _panel_axes(gs[1])
        _plot_with_hists(
            ax_main_r,
            ax_histx_r,
            ax_histy_r,
            add_codebook=True,
            panel_title="latents + codebook",
        )
        ax_main_r.set_xlim(xlim)
        ax_main_r.set_ylim(ylim)

        fig.suptitle(title)
        # out_file = out_folder / f"{filestem}.png"
        return fig
        # fig.savefig(out_file, dpi=200)
        # plt.close(fig)
        # print("Wrote PCA plot to", out_file)

    # Prepare PC1-2
    X_pca2 = X_pca_full[:, :2]
    cb_pca_full = None
    cb_pca2 = None
    if codebook_vectors is not None:
        try:
            cb_pca_full = pca.transform(codebook_vectors)
            cb_pca2 = cb_pca_full[:, :2]
        except Exception as e:
            print("Failed to project codebook into PCA space:", e)

    if codebook_vectors is not None:
        filestem2 = f"latent_pca2_codebook{suf}"
        codebook_name2 = f"codebook_pca2{suf}"
    else:
        filestem2 = f"latent_pca2{suf}"
        codebook_name2 = None

    fig12 =_plot_pca_pair(
        X_pca2,
        cb_pca2,
        filestem2,
        codebook_name2,
        "PC 1",
        "PC 2",
        "Top-2 PCA of encoder latents (pre-quant) with codebook overlay",
    )
    return fig12

class VocosExp(pl.LightningModule):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: Backbone,
        head: FourierHead,
        resume_config: str,
        resume_model: str,
        sample_rate: int = 24000,
        initial_learning_rate: float = 2e-4,
        num_warmup_steps: int = 0,
        mel_loss_coeff: float = 45,
        mrd_loss_coeff: float = 1.0,
        pretrain_mel_steps: int = 0,
        decay_mel_coeff: bool = False,
        evaluate_utmos: bool = False,
        evaluate_pesq: bool = False,
        evaluate_periodicty: bool = False,
        resume: bool = False,
        use_discriminator: bool = True,
        use_nerd: bool = False,
        train_nerd_only: bool = False,
        commit_loss_coef: float = 10.0,
        respawn_on_nerd_update: bool = False,
    ):
        """
        Args:
            feature_extractor (FeatureExtractor): An instance of FeatureExtractor to extract features from audio signals.
            backbone (Backbone): An instance of Backbone model.
            head (FourierHead):  An instance of Fourier head to generate spectral coefficients and reconstruct a waveform.
            sample_rate (int): Sampling rate of the audio signals.
            initial_learning_rate (float): Initial learning rate for the optimizer.
            num_warmup_steps (int): Number of steps for the warmup phase of learning rate scheduler. Default is 0.
            mel_loss_coeff (float, optional): Coefficient for Mel-spectrogram loss in the loss function. Default is 45.
            mrd_loss_coeff (float, optional): Coefficient for Multi Resolution Discriminator loss. Default is 1.0.
            pretrain_mel_steps (int, optional): Number of steps to pre-train the model without the GAN objective. Default is 0.
            decay_mel_coeff (bool, optional): If True, the Mel-spectrogram loss coefficient is decayed during training. Default is False.
            evaluate_utmos (bool, optional): If True, UTMOS scores are computed for each validation run.
            evaluate_pesq (bool, optional): If True, PESQ scores are computed for each validation run.
            evaluate_periodicty (bool, optional): If True, periodicity scores are computed for each validation run.
            use_discriminator: bool = True, Whether to use discriminators during training.
            train_nerd_only: bool = False,
        """
        super().__init__()
        self.save_hyperparameters(ignore=["feature_extractor", "backbone", "head"])

        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head

        self.resume_config = resume_config
        self.resume_model = resume_model
        self.resume = resume

        self.multiperioddisc = MultiPeriodDiscriminator()
        self.multiresddisc = MultiResolutionDiscriminator()

        
        self.dac = DACDiscriminator()

        self.dacdiscriminator = DACGANLoss(self.dac)

        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.melspec_loss = MelSpecReconstructionLoss(sample_rate=sample_rate)

        self.train_discriminator = False
        self.base_mel_coeff = self.mel_loss_coeff = mel_loss_coeff

        # self.nerd_config = nerd_config
        # """
        # RuntimeError: Training with multiple optimizers is only supported with manual optimization. 
        # Set `self.automatic_optimization = False`, then access your optimizers in `training_step` 
        # with `opt1, opt2, ... = self.optimizers()`.
        # """
        # self.automatic_optimization = False

    def configure_optimizers(self):
        disc_params = [
            {"params": self.multiperioddisc.parameters()},
            {"params": self.multiresddisc.parameters()},
            {"params": self.dac.parameters()},
        ]
        gen_params = [
            {"params": self.feature_extractor.parameters()},
            {"params": self.backbone.parameters()},
            {"params": self.head.parameters()},
        ]

        opt_disc = torch.optim.AdamW(disc_params, lr=self.hparams.initial_learning_rate)
        opt_gen = torch.optim.AdamW(gen_params, lr=self.hparams.initial_learning_rate)

        max_steps = self.trainer.max_steps // 2  # Max steps per optimizer
        scheduler_disc = transformers.get_cosine_schedule_with_warmup(
            opt_disc, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=max_steps,
        )
        scheduler_gen = transformers.get_cosine_schedule_with_warmup(
            opt_gen, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=max_steps,
        )

        if self.hparams.use_nerd:
            codebook = self.feature_extractor.encodec.quantizer.vq.layers[0]._codebook
            nerd_params = [
                {"params": codebook.nerd_sampler.dec.parameters()}
            ]
            opt_nerd = torch.optim.AdamW(nerd_params, lr=codebook.nerd_config.lr)
            scheduler_nerd = transformers.get_cosine_schedule_with_warmup(
                opt_nerd, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=max_steps,
            )

            return (
                [opt_nerd, opt_disc, opt_gen],
                [
                    {"scheduler": scheduler_nerd, "interval": "step"},
                    {"scheduler": scheduler_disc, "interval": "step"},
                    {"scheduler": scheduler_gen, "interval": "step"},
                ],
            )

        return (
            [opt_disc, opt_gen],
            [
                {"scheduler": scheduler_disc, "interval": "step"},
                {"scheduler": scheduler_gen, "interval": "step"},
            ],
        )

    def forward(self, audio_input, **kwargs):
        features, _, commit_loss = self.feature_extractor(audio_input, **kwargs)
        # print('1111', self.feature_extractor.state_dict()['encodec.decoder.model.3.convtr.convtr.weight_g'])
        x = self.backbone(features, **kwargs)
        audio_output = self.head(x)
        return audio_output, commit_loss

    def training_step(self, batch, batch_idx, optimizer_idx, **kwargs):
        audio_input = batch

        if self.hparams.use_nerd and self.hparams.train_nerd_only and optimizer_idx != 0:
            return None

        if optimizer_idx == 0 and self.hparams.use_nerd:
            with torch.no_grad():
                # print(f"{audio_input.shape=}")
                audio_input = audio_input.unsqueeze(1)
                features = self.feature_extractor.encodec.encoder(audio_input)
                # print(f"{features.shape=}")
                features = rearrange(features, "b d n -> b n d")
                features = self.feature_extractor.encodec.quantizer.vq.layers[0].project_in(features)
                # print(f"{features.shape=}")
                # (B, C, T) to # (B*T, C)
                B, T, C = features.shape
                features = features.contiguous().view(B * T, C)
                # 40 * 225 = 9000
                # print(f"{features.shape=}")
            features = features.detach()
            nerd_sampler = self.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.nerd_sampler
            nerd_sampler.add_latents(features)
            # print(f"{next(nerd_sampler.dec.parameters()).device=}")
            nerd_loss = nerd_sampler._train_step()
            self.log("nerd/nerd_loss", nerd_loss, on_step=True, on_epoch=False, prog_bar=True)
            self.log("nerd/sigma", nerd_sampler.dec.sigma, on_step=True, on_epoch=False, prog_bar=True)

            if (batch_idx+1) % 1000 == 0:
                features = features.cpu().numpy()
                nerd_codebook = nerd_sampler.sample(1024)
                fig = plot_pca_components(features, nerd_codebook.cpu().numpy(), 42)
                
                self.logger.experiment.add_figure(
                    f"nerd_latent_space/pca_step_{self.global_step}", fig, global_step=self.global_step
                )
                self.logger.experiment.flush()

                # codebook = self.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.embed.cpu().numpy()
                # fig_cb = plot_pca_components(features, codebook, 42)
                # self.logger.experiment.add_figure(
                #     f"codebook_latent_space/pca_step_{self.global_step}", fig_cb, global_step=self.global_step
                # )
            
            if self.hparams.respawn_on_nerd_update:
                # respawn dead codewords
                codebook = self.feature_extractor.encodec.quantizer.vq.layers[0]._codebook
                codebook.replace_all_with_nerd()
            # self.log("nerd/commit_loss", commit_loss, prog_bar=True)
            # total_nerd_loss = nerd_loss + 1000 * commit_loss
            return nerd_loss if nerd_sampler.buf.full else None #torch.tensor(0.0, device=self.device)

        # train discriminator
        discriminator_optimizer_idx = 1 if self.hparams.use_nerd else 0
        if optimizer_idx == discriminator_optimizer_idx and self.train_discriminator:
            # opt, _ = self.optimizers()
            # opt.zero_grad()
            with torch.no_grad():
                audio_hat, _ = self(audio_input, **kwargs)


            loss_dac=self.dacdiscriminator.discriminator_loss(audio_hat.unsqueeze(1),audio_input.unsqueeze(1))

            real_score_mp, gen_score_mp, _, _ = self.multiperioddisc(y=audio_input, y_hat=audio_hat, **kwargs,)
            real_score_mrd, gen_score_mrd, _, _ = self.multiresddisc(y=audio_input, y_hat=audio_hat, **kwargs,)
            loss_mp, loss_mp_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mp, disc_generated_outputs=gen_score_mp
            )
            loss_mrd, loss_mrd_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mrd, disc_generated_outputs=gen_score_mrd
            )
            loss_mp /= len(loss_mp_real)
            loss_mrd /= len(loss_mrd_real)
            loss = loss_mp + self.hparams.mrd_loss_coeff * loss_mrd + loss_dac

            self.log("discriminator/total", loss, prog_bar=True)
            self.log("discriminator/multi_period_loss", loss_mp)
            self.log("discriminator/multi_res_loss", loss_mrd)
            self.log("discriminator/dac", loss_dac)
            # self.manual_backward(loss)
            # opt.step()
            return loss

        # train generator
        generator_optimizer_idx = 2 if self.hparams.use_nerd else 1
        if optimizer_idx == generator_optimizer_idx:
            # _, opt = self.optimizers()
            # opt.zero_grad()
            audio_hat, commit_loss = self(audio_input, **kwargs)
            # Mean squared error between prediction and reference for logging
            mse = torch.mean((audio_hat - audio_input) ** 2)
            self.log("train/mse", mse, on_step=True, prog_bar=True)
            if self.train_discriminator:

                loss_dac_1,loss_dac_2 = self.dacdiscriminator.generator_loss(audio_hat.unsqueeze(1),audio_input.unsqueeze(1))
                _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = self.multiperioddisc(
                    y=audio_input, y_hat=audio_hat, **kwargs,
                )
                _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresddisc(
                    y=audio_input, y_hat=audio_hat, **kwargs,
                )
                loss_gen_mp, list_loss_gen_mp = self.gen_loss(disc_outputs=gen_score_mp)
                loss_gen_mrd, list_loss_gen_mrd = self.gen_loss(disc_outputs=gen_score_mrd)
                loss_gen_mp = loss_gen_mp / len(list_loss_gen_mp)
                loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)
                loss_fm_mp = self.feat_matching_loss(fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp) / len(fmap_rs_mp)
                loss_fm_mrd = self.feat_matching_loss(fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd) / len(fmap_rs_mrd)

                self.log("generator/multi_period_loss", loss_gen_mp)
                self.log("generator/multi_res_loss", loss_gen_mrd)
                self.log("generator/feature_matching_mp", loss_fm_mp)
                self.log("generator/feature_matching_mrd", loss_fm_mrd)
                self.log("generator/loss_dac_1", loss_dac_1)
                self.log("generator/loss_dac_2", loss_dac_2)
            else:
                loss_gen_mp = loss_gen_mrd = loss_fm_mp = loss_fm_mrd = loss_dac_1 = loss_dac_2 = 0
            commit_loss_coef = 1000 if self.train_discriminator else self.hparams.commit_loss_coef

            mel_loss = self.melspec_loss(audio_hat, audio_input)
            loss = (
                loss_gen_mp
                + self.hparams.mrd_loss_coeff * loss_gen_mrd
                + loss_fm_mp
                + self.hparams.mrd_loss_coeff * loss_fm_mrd
                + self.mel_loss_coeff * mel_loss
                + commit_loss_coef * commit_loss
                + loss_dac_1
                + loss_dac_2
            )

            # self.log("generator/total_loss", loss, prog_bar=True)
            self.log("mel_loss_coeff", self.mel_loss_coeff, on_step=True, prog_bar=False)
            self.log("generator/mel_loss", mel_loss, on_step=True, prog_bar=False)

            self.log("quantizer/commit_loss", commit_loss, on_step=True, prog_bar=False)
            expired_codes = self.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.expired_codes
            self.log("quantizer/expired_codes", expired_codes, on_step=True, prog_bar=True)

            if (batch_idx+1) % 1000 == 0:
                with torch.no_grad():
                    # print(f"{audio_input.shape=}")
                    audio_input = audio_input.unsqueeze(1)
                    features = self.feature_extractor.encodec.encoder(audio_input)
                    # print(f"{features.shape=}")
                    features = rearrange(features, "b d n -> b n d")
                    features = self.feature_extractor.encodec.quantizer.vq.layers[0].project_in(features)
                    # print(f"{features.shape=}")
                    # (B, C, T) to # (B*T, C)
                    B, T, C = features.shape
                    features = features.contiguous().view(B * T, C)
                    # 40 * 225 = 9000
                    # print(f"{features.shape=}")
                features = features.detach().cpu().numpy()

                codebook = self.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.embed.cpu().numpy()
                fig_cb = plot_pca_components(features, codebook, 42)
                self.logger.experiment.add_figure(
                    f"codebook_latent_space_pca/step_{self.global_step}", fig_cb, global_step=self.global_step
                )

            # if self.global_step % 1000 == 0 and self.global_rank == 0:
            #     self.logger.experiment.add_audio(
            #         "train/audio_in", audio_input[0].data.cpu(), self.global_step, self.hparams.sample_rate
            #     )
            #     self.logger.experiment.add_audio(
            #         "train/audio_pred", audio_hat[0].data.cpu(), self.global_step, self.hparams.sample_rate
            #     )
            #     with torch.no_grad():
            #         mel = safe_log(self.melspec_loss.mel_spec(audio_input[0]))
            #         mel_hat = safe_log(self.melspec_loss.mel_spec(audio_hat[0]))
            #     self.logger.experiment.add_image(
            #         "train/mel_target",
            #         plot_spectrogram_to_numpy(mel.data.cpu().numpy()),
            #         self.global_step,
            #         dataformats="HWC",
            #     )
            #     self.logger.experiment.add_image(
            #         "train/mel_pred",
            #         plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()),
            #         self.global_step,
            #         dataformats="HWC",
            #     )
            #     self.logger.experiment.flush()

            # self.manual_backward(loss)
            # opt.step()
            return loss

    def on_validation_epoch_start(self):
        if self.hparams.evaluate_utmos:
            from metrics.UTMOS import UTMOSScore

            if not hasattr(self, "utmos_model"):
                self.utmos_model = UTMOSScore(device=self.device)

    def validation_step(self, batch, batch_idx, **kwargs):
        audio_input = batch
        audio_hat, commit_loss = self(audio_input, **kwargs)

        audio_16_khz = torchaudio.functional.resample(audio_input, orig_freq=self.hparams.sample_rate, new_freq=16000)
        audio_hat_16khz = torchaudio.functional.resample(audio_hat, orig_freq=self.hparams.sample_rate, new_freq=16000)

        if self.hparams.evaluate_periodicty:
            from metrics.periodicity import calculate_periodicity_metrics

            periodicity_loss, pitch_loss, f1_score = calculate_periodicity_metrics(audio_16_khz, audio_hat_16khz)
        else:
            periodicity_loss = pitch_loss = f1_score = 0

        if self.hparams.evaluate_utmos:
            utmos_score = self.utmos_model.score(audio_hat_16khz.unsqueeze(1)).mean()
        else:
            utmos_score = torch.zeros(1, device=self.device)

        if self.hparams.evaluate_pesq:
            from pesq import pesq

            pesq_score = 0
            for ref, deg in zip(audio_16_khz.cpu().numpy(), audio_hat_16khz.cpu().numpy()):
                pesq_score += pesq(16000, ref, deg, "wb", on_error=1)
            pesq_score /= len(audio_16_khz)
            pesq_score = torch.tensor(pesq_score)
        else:
            pesq_score = torch.zeros(1, device=self.device)

        mel_loss = self.melspec_loss(audio_hat.unsqueeze(1), audio_input.unsqueeze(1))
        total_loss = mel_loss + (5 - utmos_score) + (5 - pesq_score) + 1000 * commit_loss

        # Mean squared error between prediction and reference (averaged over all samples)
        # print("audio_input", audio_input.shape)
        # print("audio_hat", audio_hat.shape)
        mse = torch.mean((audio_hat - audio_input) ** 2)

        _, codes, _ = self.feature_extractor.infer(audio_input, **kwargs)
        # print(f"{codes=}")
        return {
            "val_loss": total_loss,
            "mel_loss": mel_loss,
            "utmos_score": utmos_score,
            "pesq_score": pesq_score,
            "periodicity_loss": periodicity_loss,
            "pitch_loss": pitch_loss,
            "f1_score": f1_score,
            "mse": mse,
            "codes": codes,
            "commit_loss": commit_loss,
            "audio_input": audio_input[0],
            "audio_pred": audio_hat[0],
        }

    def validation_epoch_end(self, outputs):
        if self.global_rank == 0:
            *_, audio_in, audio_pred = outputs[0].values()
            self.logger.experiment.add_audio(
                "val_in", audio_in.data.cpu().numpy(), self.global_step, self.hparams.sample_rate
            )
            self.logger.experiment.add_audio(
                "val_pred", audio_pred.data.cpu().numpy(), self.global_step, self.hparams.sample_rate
            )
            mel_target = safe_log(self.melspec_loss.mel_spec(audio_in))
            mel_hat = safe_log(self.melspec_loss.mel_spec(audio_pred))
            self.logger.experiment.add_image(
                "val_mel_target",
                plot_spectrogram_to_numpy(mel_target.data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            self.logger.experiment.add_image(
                "val_mel_hat",
                plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        mel_loss = torch.stack([x["mel_loss"] for x in outputs]).mean()
        utmos_score = torch.stack([x["utmos_score"] for x in outputs]).mean()
        pesq_score = torch.stack([x["pesq_score"] for x in outputs]).mean()
        periodicity_loss = np.array([x["periodicity_loss"] for x in outputs]).mean()
        pitch_loss = np.array([x["pitch_loss"] for x in outputs]).mean()
        f1_score = np.array([x["f1_score"] for x in outputs]).mean()
        mse = torch.stack([x["mse"] for x in outputs]).mean()
        commit_loss = torch.stack([x["commit_loss"] for x in outputs]).mean()

        codes = torch.stack([x["codes"] for x in outputs]).detach()
        codebook_size = self.feature_extractor.encodec.quantizer.vq.layers[0].codebook_size
        flat_codes = codes.reshape(-1).long()
        counts = torch.bincount(flat_codes, minlength=codebook_size).float()
        usage = (counts > 0).float().sum() / codebook_size
        self.log("val/codebook_usage", usage, sync_dist=True)
        probs = counts / counts.sum().clamp_min(1.0)
        probs = probs[probs > 0]
        entropy = -(probs * torch.log2(probs)).sum()
        perplexity = torch.exp2(entropy)

        self.log("val/codebook_entropy", entropy, sync_dist=True)
        self.log("val/codebook_perplexity", perplexity, sync_dist=True)

        self.log("val_loss", avg_loss, sync_dist=True)
        self.log("val/mel_loss", mel_loss, sync_dist=True)
        self.log("val/utmos_score", utmos_score, sync_dist=True)
        self.log("val/pesq_score", pesq_score, sync_dist=True)
        self.log("val/periodicity_loss", periodicity_loss, sync_dist=True)
        self.log("val/pitch_loss", pitch_loss, sync_dist=True)
        self.log("val/f1_score", f1_score, sync_dist=True)
        self.log("val/mse", mse, sync_dist=True)
        self.log("val/commit_loss", commit_loss, sync_dist=True)

    @property
    def global_step(self):
        """
        Override global_step so that it returns the total number of batches processed
        """
        return self.trainer.fit_loop.epoch_loop.total_batch_idx

    def on_train_batch_start(self, *args):
        if not self.hparams.use_discriminator:
            self.train_discriminator = False
            return
        if self.global_step >= self.hparams.pretrain_mel_steps:
            self.train_discriminator = True
        else:
            self.train_discriminator = False

    def on_train_batch_end(self, *args):
        def mel_loss_coeff_decay(current_step, num_cycles=0.5):
            max_steps = self.trainer.max_steps // 2
            if current_step < self.hparams.num_warmup_steps:
                return 1.0
            progress = float(current_step - self.hparams.num_warmup_steps) / float(
                max(1, max_steps - self.hparams.num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        if self.hparams.decay_mel_coeff:
            self.mel_loss_coeff = self.base_mel_coeff * mel_loss_coeff_decay(self.global_step + 1)


class WavTokenizer(VocosExp):
    """
    WavTokenizer is a subclass of VocosExp that overrides the parent experiment to function as a conditional GAN.
    It manages an additional `bandwidth_id` attribute, which denotes a learnable embedding corresponding to
    a specific bandwidth value of EnCodec. During training, a random bandwidth_id is generated for each step,
    while during validation, a fixed bandwidth_id is used.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: Backbone,
        head: FourierHead,
        resume_config: str,
        resume_model: str,
        sample_rate: int = 24000,
        initial_learning_rate: float = 2e-4,
        num_warmup_steps: int = 0,
        mel_loss_coeff: float = 45,
        mrd_loss_coeff: float = 1.0,
        pretrain_mel_steps: int = 0,
        decay_mel_coeff: bool = False,
        evaluate_utmos: bool = False,
        evaluate_pesq: bool = False,
        evaluate_periodicty: bool = False,
        resume: bool = False,
        use_discriminator: bool = True,
        use_nerd: bool = False,
        train_nerd_only: bool = False,
        # nerd_initial_learning_rate: float = 5e-4,
        commit_loss_coef: float = 10.0,
        respawn_on_nerd_update: bool = False,
        # nerd_config: NERDConfig = None,
    ):
        super().__init__(
            feature_extractor,
            backbone,
            head,
            resume_config,
            resume_model,
            sample_rate,
            initial_learning_rate,
            num_warmup_steps,
            mel_loss_coeff,
            mrd_loss_coeff,
            pretrain_mel_steps,
            decay_mel_coeff,
            evaluate_utmos,
            evaluate_pesq,
            evaluate_periodicty,
            resume,
            use_discriminator,
            use_nerd,
            train_nerd_only,
            # nerd_initial_learning_rate,
            commit_loss_coef,
            respawn_on_nerd_update,
            # nerd_config,
        )
        # Override with conditional discriminators
        # VocosExp.__init__(self, feature_extractor, backbone, head, resume_config, resume_model)
        # if self.resume:
        #     VocosExp.load_from_checkpoint(self.resume_model)
        self.multiperioddisc = MultiPeriodDiscriminator(num_embeddings=len(self.feature_extractor.bandwidths))
        self.multiresddisc = MultiResolutionDiscriminator(num_embeddings=len(self.feature_extractor.bandwidths))
        self.dac = DACDiscriminator()
        if self.resume:
            print('加载预训练模型:', self.resume_model)
            # with open(self.resume_config, "r") as f:
            #     config = yaml.safe_load(f)
            # feature_extractor = instantiate_class(args=(), init=config['model']['init_args']["feature_extractor"])
            # backbone = instantiate_class(args=(), init=config['model']['init_args']["backbone"])
            # head = instantiate_class(args=(), init=config['model']['init_args']["head"])

            # 不加载量化器部分权重
            state_dict_raw = torch.load(self.resume_model, map_location=self.device)['state_dict']
            state_dict_fa_qa = dict()
            state_dict_fa_en = dict()
            state_dict_fa_de = dict()
            state_dict_bb = dict()
            state_dict_hd = dict()
            state_dict_mp = dict()
            state_dict_mr = dict()
            state_dict_dac = dict()
            for k, v in state_dict_raw.items():
                # breakpoint()
                if k.startswith('feature_extractor.encodec.quantizer'):
                    # breakpoint()
                    # print("*****",k)
                    ss = k[46:48]
                    if ss[-1] == '.':
                        num = int(ss[0])
                        # print("num,k",num,k[36:])
                        if num <= 7:
                            state_dict_fa_qa[k[36:]] = v
                if k.startswith('feature_extractor.encodec.encoder'):
                    state_dict_fa_en[k[34:]] = v
                if k.startswith('feature_extractor.encodec.decoder'):
                    state_dict_fa_de[k[34:]] = v
                if k.startswith('backbone.'):
                    state_dict_bb[k[9:]] = v
                if k.startswith('head.'):
                    state_dict_hd[k[5:]] = v
                if k.startswith('multiperioddisc.'):
                    state_dict_mp[k[16:]] = v
                if k.startswith('multiresddisc.'):
                    state_dict_mr[k[14:]] = v
                if k.startswith('dac.'):
                    state_dict_dac[k[4:]] = v
            # breakpoint()
            # feature_extractor.encodec.quantizer.load_state_dict(state_dict_fa_qa, strict=True)
            feature_extractor.encodec.encoder.load_state_dict(state_dict_fa_en, strict=True)
            feature_extractor.encodec.decoder.load_state_dict(state_dict_fa_de, strict=True)
            feature_extractor.encodec.quantizer.load_state_dict(state_dict_fa_qa, strict=False)
            backbone.load_state_dict(state_dict_bb, strict=True)
            head.load_state_dict(state_dict_hd, strict=True)
            self.feature_extractor = feature_extractor.to(self.device)
            self.backbone = backbone.to(self.device)
            self.head = head.to(self.device)
            self.multiperioddisc.load_state_dict(state_dict_mp, strict=True)
            self.multiresddisc.load_state_dict(state_dict_mr, strict=True)
            self.dac.load_state_dict(state_dict_dac, strict=True)

    def training_step(self, *args):
        # print('-------------------train--------------------')
        # if self.global_rank == 0 and self.resume:
        #     config_path = self.resume_config
        #     model_path = self.resume_model
        #     self.pretrained_load(config_path, model_path)
        #     print('加载预训练模型:', model_path)
        bandwidth_id = torch.randint(low=0, high=len(self.feature_extractor.bandwidths), size=(1,), device=self.device,)
        output = super().training_step(*args, bandwidth_id=bandwidth_id)
        return output

    def validation_step(self, *args):
        # print('-------------------valid--------------------')
        bandwidth_id = torch.tensor([0], device=self.device)
        output = super().validation_step(*args, bandwidth_id=bandwidth_id)
        return output

    def validation_epoch_end(self, outputs):
        if self.global_rank == 0:
            *_, audio_in, _ = outputs[0].values()
            # Resynthesis with encodec for reference
            self.feature_extractor.encodec.set_target_bandwidth(self.feature_extractor.bandwidths[0])
            encodec_audio = self.feature_extractor.encodec(audio_in[None, None, :])
            self.logger.experiment.add_audio(
                "encodec", encodec_audio[0, 0].data.cpu().numpy(), self.global_step, self.hparams.sample_rate,
            )

        super().validation_epoch_end(outputs)
