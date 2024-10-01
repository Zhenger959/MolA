'''
Author: Jiaxin Zheng
Date: 2023-09-01 15:17:19
LastEditors: Jiaxin Zheng
LastEditTime: 2024-10-01 19:38:27
Description: 
'''
import torch
import torch.nn as nn

from onmt.decoders.decoder import DecoderBase

from onmt.modules import MultiHeadedAttention

from onmt.modules import AverageAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.modules.position_ffn import ActivationFunction
from onmt.utils.misc import sequence_mask
from transformers.models.t5.modeling_t5 import T5Stack
from transformers import T5ForConditionalGeneration

from onmt.encoders.encoder import EncoderBase
from src.models.components.encoder_mha import MultiHeadedAttention as MHA  # TransformerEncoder
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.modules.position_ffn import ActivationFunction
from onmt.utils.misc import sequence_mask

class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
        pos_ffn_activation_fn (ActivationFunction):
            activation function choice for PositionwiseFeedForward layer
        add_qkvbias (bool): whether to add bias to the Key/Value nn.Linear
        num_kv (int): number of heads for KV when different vs Q (multiquery)
        add_ffnbias (bool): whether to add bias to the FF nn.Linear
        parallel_residual (bool): Use parallel residual connections in each layer block, as used
            by the GPT-J and GPT-NeoX models
        layer_norm (string): type of layer normalization standard/rms
        norm_eps (float): layer norm epsilon
        use_ckpting (List): layers for which we checkpoint for backward
        parallel_gpu (int): Number of gpu for tensor parallelism
        rotary_interleave (bool): Interleave the head dimensions when rotary
            embeddings are applied
        rotary_theta (int): rotary base theta
    """

    def __init__(
        self,
        d_model,
        heads,
        d_ff,
        dropout,
        attention_dropout,
        max_relative_positions=0,
        relative_positions_buckets=0,
        pos_ffn_activation_fn=ActivationFunction.relu,
        add_qkvbias=False,
        num_kv=0,
        add_ffnbias=True,
        parallel_residual=False,
        layer_norm="standard",
        norm_eps=1e-6,
        use_ckpting=[],
        parallel_gpu=1,
        rotary_interleave=True,
        rotary_theta=1e4,
    ):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MHA(
            heads,
            d_model,
            dropout=attention_dropout,
            is_decoder=False,
            max_relative_positions=max_relative_positions,
            relative_positions_buckets=relative_positions_buckets,
            rotary_interleave=rotary_interleave,
            rotary_theta=rotary_theta,
            attn_type="self",
            add_qkvbias=add_qkvbias,
            num_kv=num_kv,
            use_ckpting=use_ckpting,
            parallel_gpu=parallel_gpu,
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model,
            d_ff,
            dropout,
            pos_ffn_activation_fn,
            # add_ffnbias,
            # parallel_residual,
            # layer_norm,
            # norm_eps,
            # use_ckpting=use_ckpting,
            # parallel_gpu=parallel_gpu,
        )
        self.parallel_residual = parallel_residual
        if layer_norm == "standard":
            self.layer_norm = nn.LayerNorm(d_model, eps=norm_eps)
        elif layer_norm == "rms":
            self.layer_norm = RMSNorm(d_model, eps=norm_eps)
        else:
            raise ValueError(f"{layer_norm} layer norm type is not supported")
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, layer_in, mask):
        """
        Args:
            layer_in (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):
            * layer_out ``(batch_size, src_len, model_dim)``
        """
        norm_layer_in = self.layer_norm(layer_in)
        context, _ = self.self_attn(
            norm_layer_in, norm_layer_in, norm_layer_in, mask=mask
        )
        if self.dropout_p > 0:
            context = self.dropout(context)
        if self.parallel_residual:
            # feed_forward applies residual, so we remove and apply residual with un-normed
            layer_out = (
                self.feed_forward(norm_layer_in) - norm_layer_in + layer_in + context
            )
        else:
            layer_out = context + layer_in
            layer_out = self.feed_forward(layer_out)

        return layer_out

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout

class TransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
        pos_ffn_activation_fn (ActivationFunction):
            activation function choice for PositionwiseFeedForward layer

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * enc_out ``(batch_size, src_len, model_dim)``
        * encoder final state: None in the case of Transformer
        * src_len ``(batch_size)``
    """

    def __init__(
        self,
        num_layers,
        d_model,
        heads,
        d_ff,
        dropout,
        attention_dropout,
        # embeddings,
        max_relative_positions,
        relative_positions_buckets=0,
        pos_ffn_activation_fn=ActivationFunction.relu,
        add_qkvbias=False,
        num_kv=0,
        add_ffnbias=True,
        parallel_residual=False,
        layer_norm="standard",
        norm_eps=1e-6,
        use_ckpting=[],
        parallel_gpu=1,
        rotary_interleave=True,
        rotary_theta=1e4,
    ):
        super(TransformerEncoder, self).__init__()

        # self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    heads,
                    d_ff,
                    dropout,
                    attention_dropout,
                    max_relative_positions=max_relative_positions,
                    relative_positions_buckets=relative_positions_buckets,
                    pos_ffn_activation_fn=pos_ffn_activation_fn,
                    add_qkvbias=add_qkvbias,
                    num_kv=num_kv,
                    add_ffnbias=add_ffnbias,
                    parallel_residual=parallel_residual,
                    layer_norm=layer_norm,
                    norm_eps=norm_eps,
                    use_ckpting=use_ckpting,
                    parallel_gpu=parallel_gpu,
                    rotary_interleave=rotary_interleave,
                    rotary_theta=rotary_theta,
                )
                for i in range(num_layers)
            ]
        )
        if layer_norm == "standard":
            self.layer_norm = nn.LayerNorm(d_model, eps=norm_eps)
        elif layer_norm == "rms":
            self.layer_norm = RMSNorm(d_model, eps=norm_eps)
        else:
            raise ValueError(f"{layer_norm} layer norm type is not supported")

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_hid_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0]
            if type(opt.attention_dropout) is list
            else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions,
            opt.relative_positions_buckets,
            pos_ffn_activation_fn=opt.pos_ffn_activation_fn,
            add_qkvbias=opt.add_qkvbias,
            num_kv=opt.num_kv,
            add_ffnbias=opt.add_ffnbias,
            parallel_residual=opt.parallel_residual,
            layer_norm=opt.layer_norm,
            norm_eps=opt.norm_eps,
            use_ckpting=opt.use_ckpting,
            parallel_gpu=opt.world_size
            if opt.parallel_mode == "tensor_parallel"
            else 1,
            rotary_interleave=opt.rotary_interleave,
            rotary_theta=opt.rotary_theta,
        )

    def forward(self, enc_out, src_len=None):
        """See :func:`EncoderBase.forward()`"""
        # enc_out = self.embeddings(src)
        mask = sequence_mask(src_len).unsqueeze(1).unsqueeze(1)
        mask = mask.expand(-1, -1, mask.size(3), -1)
        # Padding mask is now (batch x 1 x slen x slen)
        # 1 to be expanded to number of heads in MHA
        # Run the forward pass of every layer of the tranformer.

        for layer in self.transformer:
            enc_out = layer(enc_out, mask)
        enc_out = self.layer_norm(enc_out)
        return enc_out, None, src_len

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)

class TransformerDecoderLayerBase(nn.Module):
    def __init__(
        self,
        d_model,
        heads,
        d_ff,
        dropout,
        attention_dropout,
        self_attn_type="scaled-dot",
        ckpt_path='',
        max_relative_positions=0,
        aan_useffn=False,
        full_context_alignment=False,
        alignment_heads=0,
        pos_ffn_activation_fn=ActivationFunction.relu,
    ):
        """
        Args:
            d_model (int): the dimension of keys/values/queries in
                :class:`MultiHeadedAttention`, also the input size of
                the first-layer of the :class:`PositionwiseFeedForward`.
            heads (int): the number of heads for MultiHeadedAttention.
            d_ff (int): the second-layer of the
                :class:`PositionwiseFeedForward`.
            dropout (float): dropout in residual, self-attn(dot) and
                feed-forward
            attention_dropout (float): dropout in context_attn  (and
                self-attn(avg))
            self_attn_type (string): type of self-attention scaled-dot,
                average
            max_relative_positions (int):
                Max distance between inputs in relative positions
                representations
            aan_useffn (bool): Turn on the FFN layer in the AAN decoder
            full_context_alignment (bool):
                whether enable an extra full context decoder forward for
                alignment
            alignment_heads (int):
                N. of cross attention heads to use for alignment guiding
            pos_ffn_activation_fn (ActivationFunction):
                activation function choice for PositionwiseFeedForward layer

        """
        super(TransformerDecoderLayerBase, self).__init__()

        if self_attn_type == "scaled-dot":
            self.self_attn = MultiHeadedAttention(
                heads,
                d_model,
                dropout=attention_dropout,
                max_relative_positions=max_relative_positions,
            )
        elif self_attn_type == "average":
            self.self_attn = AverageAttention(
                d_model, dropout=attention_dropout, aan_useffn=aan_useffn
            )
        elif self_attn_type == "MolT5":
            self.self_attn = T5ForConditionalGeneration.from_pretrained(ckpt_path).encoder
            encoder_d_model=self.self_attn.config.d_model
            # self.input_linear=nn.Linear(d_model,encoder_d_model)
            # self.output_linear=nn.Linear(encoder_d_model,d_model)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout,
                                                    pos_ffn_activation_fn
                                                    )
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        self.full_context_alignment = full_context_alignment
        self.alignment_heads = alignment_heads

    def forward(self, *args, **kwargs):
        """Extend `_forward` for (possibly) multiple decoder pass:
        Always a default (future masked) decoder forward pass,
        Possibly a second future aware decoder pass for joint learn
        full context alignement, :cite:`garg2019jointly`.

        Args:
            * All arguments of _forward.
            with_align (bool): whether return alignment attention.

        Returns:
            (FloatTensor, FloatTensor, FloatTensor or None):

            * output ``(batch_size, T, model_dim)``
            * top_attn ``(batch_size, T, src_len)``
            * attn_align ``(batch_size, T, src_len)`` or None
        """
        with_align = kwargs.pop("with_align", False)
        output, attns = self._forward(*args, **kwargs)
        top_attn = attns[:, 0, :, :].contiguous()
        attn_align = None
        if with_align:
            if self.full_context_alignment:
                # return _, (B, Q_len, K_len)
                _, attns = self._forward(*args, **kwargs, future=True)

            if self.alignment_heads > 0:
                attns = attns[:, : self.alignment_heads, :, :].contiguous()
            # layer average attention across heads, get ``(B, Q, K)``
            # Case 1: no full_context, no align heads -> layer avg baseline
            # Case 2: no full_context, 1 align heads -> guided align
            # Case 3: full_context, 1 align heads -> full cte guided align
            attn_align = attns.mean(dim=1)
        return output, top_attn, attn_align

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.drop.p = dropout

    def _forward(self, *args, **kwargs):
        raise NotImplementedError

    def _compute_dec_mask(self, tgt_pad_mask, future):
        tgt_len = tgt_pad_mask.size(-1)
        if not future:  # apply future_mask, result mask in (B, T, T)
            future_mask = torch.ones(
                [tgt_len, tgt_len],
                device=tgt_pad_mask.device,
                dtype=torch.uint8,
            )
            future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
            # BoolTensor was introduced in pytorch 1.2
            try:
                future_mask = future_mask.bool()
            except AttributeError:
                pass
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)
        else:  # only mask padding, result mask in (B, 1, T)
            dec_mask = tgt_pad_mask
        return dec_mask

    def _forward_self_attn(self, inputs_norm, dec_mask, layer_cache, step):
        """
        Returns: ()
        """
        if isinstance(self.self_attn, MultiHeadedAttention):
            return self.self_attn(
                inputs_norm,
                inputs_norm,
                inputs_norm,
                mask=dec_mask,
                layer_cache=layer_cache,
                attn_type="self",
            )  # [2,114,256] [2,8,114,114]
        elif isinstance(self.self_attn, T5Stack):
            # inputs_norm=self.input_linear(inputs_norm)
            outputs=self.self_attn(inputs_embeds=inputs_norm,attention_mask=dec_mask)['last_hidden_state']
            # outputs=self.output_linear(outputs)
            return (outputs,torch.Tensor([0]).to(inputs_norm.device))
        
        elif isinstance(self.self_attn, AverageAttention):
            return self.self_attn(
                inputs_norm, mask=dec_mask, layer_cache=layer_cache, step=step
            )
        else:
            raise ValueError(
                f"self attention {type(self.self_attn)} not supported"
            )

class TransformerDecoderLayer(TransformerDecoderLayerBase):
    """Transformer Decoder layer block in Pre-Norm style.
    Pre-Norm style is an improvement w.r.t. Original paper's Post-Norm style,
    providing better converge speed and performance. This is also the actual
    implementation in tensor2tensor and also avalable in fairseq.
    See https://tunz.kr/post/4 and :cite:`DeeperTransformer`.

    .. mermaid::

        graph LR
        %% "*SubLayer" can be self-attn, src-attn or feed forward block
            A(input) --> B[Norm]
            B --> C["*SubLayer"]
            C --> D[Drop]
            D --> E((+))
            A --> E
            E --> F(out)

    """

    def __init__(
        self,
        d_model,
        heads,
        d_ff,
        dropout,
        attention_dropout,
        self_attn_type="scaled-dot",
        ckpt_path='',
        max_relative_positions=0,
        aan_useffn=False,
        full_context_alignment=False,
        alignment_heads=0,
        pos_ffn_activation_fn=ActivationFunction.relu,
    ):
        """
        Args:
            See TransformerDecoderLayerBase
        """
        super(TransformerDecoderLayer, self).__init__(
            d_model,
            heads,
            d_ff,
            dropout,
            attention_dropout,
            self_attn_type,
            ckpt_path,
            max_relative_positions,
            aan_useffn,
            full_context_alignment,
            alignment_heads,
            pos_ffn_activation_fn=pos_ffn_activation_fn,
        )
        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout
        )
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)

    def update_dropout(self, dropout, attention_dropout):
        super(TransformerDecoderLayer, self).update_dropout(
            dropout, attention_dropout
        )
        self.context_attn.update_dropout(attention_dropout)

    def _forward(
        self,
        inputs,
        memory_bank,
        src_pad_mask,
        tgt_pad_mask,
        layer_cache=None,
        step=None,
        future=False,
    ):
        """A naive forward pass for transformer decoder.

        # T: could be 1 in the case of stepwise decoding or tgt_len

        Args:
            inputs (FloatTensor): ``(batch_size, T, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (bool): ``(batch_size, 1, src_len)``
            tgt_pad_mask (bool): ``(batch_size, 1, T)``
            layer_cache (dict or None): cached layer info when stepwise decode
            step (int or None): stepwise decoding counter
            future (bool): If set True, do not apply future_mask.

        Returns:
            (FloatTensor, FloatTensor):

            * output ``(batch_size, T, model_dim)``
            * attns ``(batch_size, head, T, src_len)``

        """
        dec_mask = None

        if inputs.size(1) > 1:
            # masking is necessary when sequence length is greater than one
            dec_mask = self._compute_dec_mask(tgt_pad_mask, future)

        inputs_norm = self.layer_norm_1(inputs)

        query, _ = self._forward_self_attn(
            inputs_norm, dec_mask, layer_cache, step
        )

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid, attns = self.context_attn(
            memory_bank,
            memory_bank,
            query_norm,
            mask=src_pad_mask,
            layer_cache=layer_cache,
            attn_type="context",
        )
        output = self.feed_forward(self.drop(mid) + query)

        return output, attns


class TransformerDecoderBase(DecoderBase):
    def __init__(self, d_model, copy_attn, alignment_layer):
        super(TransformerDecoderBase, self).__init__()

        # Decoder State
        self.state = {}

        # previously, there was a GlobalAttention module here for copy
        # attention. But it was never actually used -- the "copy" attention
        # just reuses the context attention.
        self._copy = copy_attn
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.alignment_layer = alignment_layer

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.copy_attn,
            opt.self_attn_type,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout) is list else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions,
            opt.aan_useffn,
            opt.full_context_alignment,
            opt.alignment_layer,
            alignment_heads=opt.alignment_heads,
            pos_ffn_activation_fn=opt.pos_ffn_activation_fn,
        )

    def init_state(self, src, memory_bank, enc_hidden):
        """Initialize decoder state."""
        self.state["src"] = src
        self.state["cache"] = None

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        if self.state["src"] is not None:
            self.state["src"] = fn(self.state["src"], 1)
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def detach_state(self):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)


class TransformerDecoder(TransformerDecoderBase):
    """The Transformer decoder from "Attention is All You Need".
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
        num_layers (int): number of decoder layers.
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        copy_attn (bool): if using a separate copy attention
        self_attn_type (str): type of self-attention scaled-dot, average
        dropout (float): dropout in residual, self-attn(dot) and feed-forward
        attention_dropout (float): dropout in context_attn (and self-attn(avg))
        embeddings (onmt.modules.Embeddings):
            embeddings to use, should have positional encodings
        max_relative_positions (int):
            Max distance between inputs in relative positions representations
        aan_useffn (bool): Turn on the FFN layer in the AAN decoder
        full_context_alignment (bool):
            whether enable an extra full context decoder forward for alignment
        alignment_layer (int): N° Layer to supervise with for alignment guiding
        alignment_heads (int):
            N. of cross attention heads to use for alignment guiding
    """

    def __init__(
        self,
        num_layers,
        d_model,
        heads,
        d_ff,
        copy_attn,
        self_attn_type,
        ckpt_path,
        dropout,
        attention_dropout,
        max_relative_positions,
        aan_useffn,
        full_context_alignment,
        alignment_layer,
        alignment_heads,
        pos_ffn_activation_fn=ActivationFunction.relu,
    ):
        super(TransformerDecoder, self).__init__(
            d_model, copy_attn, alignment_layer
        )

        self.transformer_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model,
                    heads,
                    d_ff,
                    dropout,
                    attention_dropout,
                    self_attn_type=self_attn_type,
                    ckpt_path=ckpt_path,
                    max_relative_positions=max_relative_positions,
                    aan_useffn=aan_useffn,
                    full_context_alignment=full_context_alignment,
                    alignment_heads=alignment_heads,
                    pos_ffn_activation_fn=pos_ffn_activation_fn,
                )
                for i in range(num_layers)
            ]
        )

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()

    def forward(self, tgt_emb, memory_bank, src_pad_mask=None, tgt_pad_mask=None, step=None, **kwargs):
        """Decode, possibly stepwise."""
        if step == 0:
            self._init_cache(memory_bank)

        batch_size, src_len, src_dim = memory_bank.size()
        device = memory_bank.device
        if src_pad_mask is None:
            src_pad_mask = torch.zeros((batch_size, 1, src_len), dtype=torch.bool, device=device)
        output = tgt_emb
        batch_size, tgt_len, tgt_dim = tgt_emb.size()
        if tgt_pad_mask is None:
            tgt_pad_mask = torch.zeros((batch_size, 1, tgt_len), dtype=torch.bool, device=device)

        future = kwargs.pop("future", False)
        with_align = kwargs.pop("with_align", False)
        attn_aligns = []
        hiddens = []

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = (
                self.state["cache"]["layer_{}".format(i)]
                if step is not None
                else None
            )
            # 在Inference的时候attn_align其实是None
            output, attn, attn_align = layer(
                output,
                memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache=layer_cache,
                step=step,
                with_align=with_align,
                future=future
            )
            hiddens.append(output)
            if attn_align is not None:
                attn_aligns.append(attn_align)

        output = self.layer_norm(output)  # (B, L, D)

        attns = {"std": attn}
        if self._copy:
            attns["copy"] = attn
        if with_align:
            attns["align"] = attn_aligns[self.alignment_layer]  # `(B, Q, K)`
            # attns["align"] = torch.stack(attn_aligns, 0).mean(0)  # All avg

        # TODO change the way attns is returned dict => list or tuple (onnx)
        return output, attns, hiddens

    def _init_cache(self, memory_bank):
        self.state["cache"] = {}
        for i, layer in enumerate(self.transformer_layers):
            layer_cache = {"memory_keys": None, "memory_values": None, "self_keys": None, "self_values": None}
            self.state["cache"]["layer_{}".format(i)] = layer_cache


class TransformerDecoderMainBase(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.enc_trans_layer = nn.Sequential(
            nn.Linear(args.encoder_dim, args.dec_hidden_size)
            # nn.LayerNorm(args.dec_hidden_size, eps=1e-6)
        )
        self.enc_pos_emb = nn.Embedding(144, args.encoder_dim) if args.enc_pos_emb else None

        self.decoder = TransformerDecoder(
            num_layers=args.dec_num_layers,
            d_model=args.dec_hidden_size,
            heads=args.dec_attn_heads,
            d_ff=args.dec_hidden_size * 4,
            copy_attn=False,
            self_attn_type=self.args.self_attn_type,
            ckpt_path=self.args.ckpt_path,
            dropout=args.hidden_dropout,
            attention_dropout=args.attn_dropout,
            max_relative_positions=args.max_relative_positions,
            aan_useffn=False,
            full_context_alignment=False,
            alignment_layer=0,
            alignment_heads=0,
            pos_ffn_activation_fn='gelu'
        )

    def enc_transform(self, encoder_out):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        max_len = encoder_out.size(1)
        device = encoder_out.device
        if self.enc_pos_emb:
            pos_emb = self.enc_pos_emb(torch.arange(max_len, device=device)).unsqueeze(0)
            encoder_out = encoder_out + pos_emb
        encoder_out = self.enc_trans_layer(encoder_out)
        return encoder_out
   
    
class TransformerMainBase(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.enc_trans_layer = nn.Sequential(
            nn.Linear(args.encoder_dim, args.dec_hidden_size)
            # nn.LayerNorm(args.dec_hidden_size, eps=1e-6)
        )
        self.enc_pos_emb = nn.Embedding(144, args.encoder_dim) if args.enc_pos_emb else None

        self.decoder = TransformerDecoder(
            num_layers=args.dec_num_layers,
            d_model=args.dec_hidden_size,
            heads=args.dec_attn_heads,
            d_ff=args.dec_hidden_size * 4,
            copy_attn=False,
            self_attn_type=self.args.self_attn_type,
            ckpt_path=self.args.ckpt_path,
            dropout=args.hidden_dropout,
            attention_dropout=args.attn_dropout,
            max_relative_positions=args.max_relative_positions,
            aan_useffn=False,
            full_context_alignment=False,
            alignment_layer=0,
            alignment_heads=0,
            pos_ffn_activation_fn='gelu'
        )
        self.encoder = TransformerEncoder(
            num_layers=args.enc_num_layers,
            d_model=args.enc_hidden_size,
            heads=args.enc_attn_heads,
            d_ff=args.enc_hidden_size * 4,
            dropout=args.hidden_dropout,
            attention_dropout=args.attn_dropout,
            max_relative_positions=args.max_relative_positions,
            pos_ffn_activation_fn='gelu'
        )

    def enc_transform(self, encoder_out):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        max_len = encoder_out.size(1)
        device = encoder_out.device
        if self.enc_pos_emb:
            pos_emb = self.enc_pos_emb(torch.arange(max_len, device=device)).unsqueeze(0)
            encoder_out = encoder_out + pos_emb

        encoder_out = self.enc_trans_layer(encoder_out)
        # encoder_out = self.encoder(encoder_out)
        return encoder_out