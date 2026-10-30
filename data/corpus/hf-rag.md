# Hugging Face Transformers: RAG

Source: https://huggingface.co/docs/transformers/main/model_doc/rag
Scope: official-documentation-excerpt; not a full-paper archive.

Transformers documentation
RAG
Transformers 🏡 View all docs AWS Trainium & Inferentia Accelerate Argilla AutoTrain Bitsandbytes CLI Chat UI Dataset viewer Datasets Deploying on AWS Diffusers Distilabel Evaluate Google Cloud Google TPUs Gradio Hub Hub Python Library Huggingface.js Inference Endpoints (dedicated) Inference Providers Kernels LeRobot Leaderboards Lighteval Microsoft Azure OpenEnv Optimum PEFT Reachy Mini Safetensors Sentence Transformers TRL Tasks Text Embeddings Inference Text Generation Inference Tokenizers Trackio Transformers Transformers.js Xet smolagents timm Search documentation main v5.21.0 v5.20.2 v5.19.0 v5.17.0 v5.15.1 v5.14.0 v5.13.1 v5.12.0 v5.11.0 v5.10.4 v5.9.0 v5.8.1 v5.7.0 v5.6.2 v5.5.4 v5.4.0 v5.3.0 v5.2.0 v5.1.0 v5.0.0 v4.57.6 v4.56.2 v4.55.4 v4.53.3 v4.52.3 v4.51.3 v4.50.0 v4.49.0 v4.48.2 v4.47.1 v4.46.3 v4.45.2 v4.44.2 v4.43.4 v4.42.4 v4.41.2 v4.40.2 v4.39.3 v4.38.2 v4.37.2 v4.36.1 v4.35.2 v4.34.1 v4.33.3 v4.32.1 v4.31.0 v4.30.0 v4.29.1 v4.28.1 v4.27.2 v4.26.1 v4.25.1 v4.24.0 v4.23.1 v4.22.2 v4.21.3 v4.20.1 v4.19.4 v4.18.0 v4.17.0 v4.16.2 v4.15.0 v4.14.1 v4.13.0 v4.12.5 v4.11.3 v4.10.1 v4.9.2 v4.8.2 v4.7.0 v4.6.0 v4.5.1 v4.4.2 v4.3.3 v4.2.2 v4.1.1 v4.0.1 v3.5.1 v3.4.0 v3.3.1 v3.2.0 v3.1.0 v3.0.2 v2.11.0 v2.10.0 v2.9.1 v2.8.0 v2.7.0 v2.6.0 v2.5.1 v2.4.1 v2.3.0 v2.2.2 v2.1.1 v2.0.0 v1.2.0 v1.1.0 v1.0.0 doc-builder-html AR DE EN ES FR HI IT JA KO PT RO TE TR ZH You are viewing main version, which requires installation from source . If you'd like
regular pip install, checkout the latest stable version ( v5.21.0 ). Join the Hugging Face community
and get access to the augmented documentation experience Collaborate on models, datasets and Spaces Faster examples with accelerated inference Switch between documentation themes Sign Up
to get started
This model was published in HF papers on 2020-05-22 and contributed to Hugging Face Transformers on 2020-11-16. Copy page
RAG
Retrieval-Augmented Generation (RAG) combines a pretrained language model (parametric memory) with access to an external data source (non-parametric memory) by means of a pretrained neural retriever. RAG fetches relevant passages and conditions its generation on them during inference. This often makes the answers more factual and lets you update knowledge by changing the index instead of retraining the whole model.
You can find all the original RAG checkpoints under the AI at Meta organization.
This model was contributed by ola13 .
Click on the RAG models in the right sidebar for more examples of how to apply RAG to different language tasks.
The examples below demonstrate how to generate text with AutoModel . AutoModel Copied
from transformers import RagRetriever, RagSequenceForGeneration, RagTokenizer
tokenizer = RagTokenizer.from_pretrained( "facebook/rag-sequence-nq" )
retriever = RagRetriever.from_pretrained(
"facebook/rag-sequence-nq" , dataset= "wiki_dpr" , index_name= "compressed"
)
model = RagSequenceForGeneration.from_pretrained(
"facebook/rag-sequence-nq" ,
retriever=retriever,
attn_implementation= "flash_attention_2" ,
device_map= "auto" ,
)
inputs = tokenizer( "How many people live in Paris?" , return_tensors= "pt" ).to(model.device)
generated = model.generate(input_ids=inputs[ "input_ids" ])
print (tokenizer.batch_decode(generated, skip_special_tokens= True )[ 0 ])
Quantization reduces memory by storing weights in lower precision. See the Quantization overview for supported backends.
The example below uses bitsandbytes to quantize the weights to 4-bits. Copied
import torch
from transformers import BitsAndBytesConfig, RagRetriever, RagSequenceForGeneration, RagTokenizer
bnb = BitsAndBytesConfig(load_in_4bit= True , bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = RagTokenizer.from_pretrained( "facebook/rag-sequence-nq" )
retriever = RagRetriever.from_pretrained(
"facebook/rag-sequence-nq" , dataset= "wiki_dpr" , index_name= "compressed"
)
model = RagSequenceForGeneration.from_pretrained(
"facebook/rag-sequence-nq" ,
retriever=retriever,
quantization_config=bnb,
device_map= "auto" ,
)
inputs = tokenizer( "How many people live in Paris?" , return_tensors= "pt" ).to(model.device)
generated = model.generate(input_ids=inputs[ "input_ids" ])
print (tokenizer.batch_decode(generated, skip_special_tokens= True )[ 0 ])
RagConfig
class transformers. RagConfig < source >
( transformers_version : str | None = None architectures : list[str] | None = None output_hidden_states : bool | None = False return_dict : bool | None = True dtype : str | torch.dtype | None = None chunk_size_feed_forward : int = 0 id2label : dict[int, str] | dict[str, str] | None = None label2id : dict[str, int] | dict[str, str] | None = None problem_type : Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None is_encoder_decoder : bool = True question_encoder : transformers.configuration_utils.PreTrainedConfig | dict | None = None generator : transformers.configuration_utils.PreTrainedConfig | dict | None = None vocab_size : int | None = None prefix : str | None = None bos_token_id : int | None = None pad_token_id : int | None = None eos_token_id : int | list[int] | None = None decoder_start_token_id : int | None = None title_sep : str = ' / ' doc_sep : str = ' // ' n_docs : int = 5 max_combined_length : int = 300 retrieval_vector_size : int = 768 retrieval_batch_size : int = 8 dataset : str = 'wiki_dpr' dataset_split : str = 'train' index_name : str = 'compressed' index_path : str | None = None passages_path : str | None = None use_dummy_dataset : bool = False reduce_loss : bool = False label_smoothing : float = 0.0 do_deduplication : bool = True exclude_bos_score : bool = False do_marginalize : bool = False output_retrieved : bool = False use_cache : bool = True dataset_revision : str | None = None )
Parameters
is_encoder_decoder ( bool , optional , defaults to True ) —
Whether the model is used as an encoder/decoder or not.
question_encoder ( PreTrainedConfig | dict , optional ) —
The config object or dictionary of the question encoder backbone
generator ( PreTrainedConfig | dict , optional ) —
The config object or dictionary of the answer generator backbone
vocab_size ( int , optional ) —
Vocabulary size of the model. Defines the number of different tokens that can be represented by the input_ids .
prefix ( str , optional ) —
A string prefix prepended to every input before passing to the generator model.
bos_token_id ( int , optional ) —
Token id used for beginning-of-stream in the vocabulary.
pad_token_id ( int , optional ) —
Token id used for padding in the vocabulary.
eos_token_id ( Union[int, list[int]] , optional ) —
Token id used for end-of-stream in the vocabulary.
decoder_start_token_id ( int , optional ) —
If an encoder-decoder model starts decoding with a different token than bos , the id of that token.
title_sep ( str , optional , defaults to " / " ) —
Separator inserted between the title and the text of the retrieved document when calling RagRetriever .
doc_sep ( str , optional , defaults to " // " ) —
Separator inserted between the text of the retrieved document and the original input when calling
RagRetriever .
n_docs ( int , optional , defaults to 5) —
Number of documents to retrieve.
max_combined_length ( int , optional , defaults to 300) —
Max length of contextualized input returned by __call__() .
retrieval_vector_size ( int , optional , defaults to 768) —
Dimensionality of the document embeddings indexed by RagRetriever .
retrieval_batch_size ( int , optional , defaults to 8) —
Retrieval batch size, defined as the number of queries issues concurrently to the faiss index encapsulated
RagRetriever .
dataset ( str , optional , defaults to "wiki_dpr" ) —
A dataset identifier of the indexed dataset in HuggingFace Datasets (list all available datasets and ids
using datasets.list_datasets() ).
dataset_split ( str , optional , defaults to "train" ) —
Which split of the dataset to load.
index_name ( str , optional , defaults to "compressed" ) —
The index name of the index associated with the dataset . One can choose between "legacy" , "exact" and
"compressed" .
index_path ( str , optional ) —
The path to the serialized faiss index on disk.
passages_path ( str , optional ) —
A path to text passages compatible with the faiss index. Required if using
LegacyIndex
use_dummy_dataset ( bool , optional , defaults to False ) —
Whether to load a “dummy” variant of the dataset specified by dataset .
reduce_loss ( bool , optional , defaults to False ) —
Whether or not to reduce the NLL loss using the torch.Tensor.sum operation.
label_smoothing ( float , optional , defaults to 0.0) —
Only relevant if return_loss is set to True . Controls the epsilon parameter value for label smoothing
in the loss calculation. If set to 0, no label smoothing is performed.
do_deduplication ( bool , optional , defaults to True ) —
Whether or not to deduplicate the generations from different context documents for a given input. Has to be
set to False if used while training with distributed backend.
exclude_bos_score ( bool , optional , defaults to False ) —
Whether or not to disregard the BOS token when computing the loss.
do_marginalize ( bool , optional , defaults to False ) —
If True , the logits are marginalized over all documents by making use of
torch.nn.functional.log_softmax .
output_retrieved ( bool , optional , defaults to False ) —
If set to True , retrieved_doc_embeds , retrieved_doc_ids , context_input_ids and
context_attention_mask are returned. See returned tensors for more detail.
use_cache ( bool , optional , defaults to True ) —
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if config.is_decoder=True or when the model is a decoder-only generative model.
dataset_revision ( str , optional ,) —
The revision (commit hash, tag, or branch) of the Hugging Face dataset used for retrieval.
This is the configuration class to store the configuration of a RagModel. It is used to instantiate a Rag
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the
Configuration objects inherit from PreTrainedConfig and can be used to control the model outputs. Read the
documentation from PreTrainedConfig for more information. from_question_encoder_generator_configs < source >
( question_encoder_config : PreTrainedConfig generator_config : PreTrainedConfig **kwargs ) → EncoderDecoderConfig
Returns
EncoderDecoderConfig
An instance of a configuration object
Instantiate a EncoderDecoderConfig (or a derived class) from a pre-trained encoder model configuration and
decoder model configuration.
RagTokenizer
class transformers. RagTokenizer < source >
( question_encoder generator )
Rag specific outputs
class transformers.models.rag.modeling_rag. RetrievAugLMMarginOutput < source >
( loss : typing.Optional[torch.FloatTensor] = None logits : typing.Optional[torch.FloatTensor] = None doc_scores : typing.Optional[torch.FloatTensor] = None past_key_values : transformers.cache_utils.Cache | None = None retrieved_doc_embeds : typing.Optional[torch.FloatTensor] = None retrieved_doc_ids : typing.Optional[torch.LongTensor] = None context_input_ids : typing.Optional[torch.LongTensor] = None context_attention_mask : typing.Optional[torch.LongTensor] = None question_encoder_last_hidden_state : typing.Optional[torch.FloatTensor] = None question_enc_hidden_states : tuple[torch.FloatTensor, ...] | None = None question_enc_attentions : tuple[torch.FloatTensor, ...] | None = None generator_enc_last_hidden_state : typing.Optional[torch.FloatTensor] = None generator_enc_hidden_states : tuple[torch.FloatTensor, ...] | None = None generator_enc_attentions : tuple[torch.FloatTensor, ...] | None = None generator_dec_hidden_states : tuple[torch.FloatTensor, ...] | None = None generator_dec_attentions : tuple[torch.FloatTensor, ...] | None = None generator_cross_attentions : tuple[torch.FloatTensor, ...] | None = None )
Parameters
loss ( torch.FloatTensor of shape (1,) , optional , returned when labels is provided) —
Language modeling loss.
logits ( torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size) ) —
Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
each vocabulary token.
doc_scores ( torch.FloatTensor of shape (batch_size, config.n_docs) ) —
Score between each retrieved document embeddings (see retrieved_doc_embeds ) and
question_encoder_last_hidden_state .
past_key_values ( Cache , optional , returned when use_cache=True is passed or when config.use_cache=True ) —
It is a Cache instance. For more details, see our kv cache guide .
Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
(see past_key_values input) to speed up sequential decoding.
retrieved_doc_embeds ( torch.FloatTensor of shape (batch_size, config.n_docs, hidden_size) , optional , returned when output_retrieved=True ) —
Embedded documents retrieved by the retriever. Is used with question_encoder_last_hidden_state to compute
the doc_scores .
retrieved_doc_ids ( torch.LongTensor of shape (batch_size, config.n_docs) , optional , returned when output_retrieved=True ) —
The indexes of the embedded documents retrieved by the retriever.
context_input_ids ( torch.LongTensor of shape (batch_size * config.n_docs, config.max_combined_length) , optional , returned when output_retrieved=True ) —
Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.
context_attention_mask ( torch.LongTensor of shape (batch_size * config.n_docs, config.max_combined_length) , optional , returned when output_retrieved=True ) —
Attention mask post-processed from the retrieved documents and the question encoder input_ids by the
retriever.
question_encoder_last_hidden_state ( torch.FloatTensor of shape (batch_size, sequence_length, hidden_size) , optional ) —
Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
model.
question_enc_hidden_states ( tuple(torch.FloatTensor) , optional , returned when output_hidden_states=True is passed or when config.output_hidden_states=True ) —
Tuple of torch.FloatTensor (one for the output of the embeddings and one for the output of each layer) of
shape (batch_size, sequence_length, hidden_size) .
Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
question_enc_attentions ( tuple(torch.FloatTensor) , optional , returned when output_attentions=True is passed or when config.output_attentions=True ) —
Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length) .
Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.
generator_enc_last_hidden_state ( torch.FloatTensor of shape (batch_size, sequence_length, hidden_size) , optional ) —
Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
generator_enc_hidden_states ( tuple(torch.FloatTensor) , optional , returned when output_hidden_states=True is passed or when config.output_hidden_states=True ) —
Tuple of torch.FloatTensor (one for the output of the embeddings and one for the output of each layer) of
shape (batch_size, sequence_length, hidden_size) .
Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
generator_enc_attentions ( tuple(torch.FloatTensor) , optional , returned when output_attentions=True is passed or when config.output_attentions=True ) —
Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length) .
Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.
generator_dec_hidden_states ( tuple(torch.FloatTensor) , optional , returned when output_hidden_states=True is passed or when config.output_hidden_states=True ) —
Tuple of torch.FloatTensor (one for the output of the embeddings and one for the output of each layer) of
shape (batch_size, sequence_length, hidden_size) .
Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
generator_dec_attentions ( tuple(torch.FloatTensor) , optional , returned when output_attentions=True is passed or when config.output_attentions=True ) —
Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length) .
Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.
generator_cross_attentions ( tuple(torch.FloatTensor) , optional , returned when output_attentions=True is passed or when config.output_attentions=True ) —
Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length) .
Cross-attentions weights of the generator decoder, after the attention softmax, used to compute the
weighted average in the cross-attention heads.
Base class for retriever augmented marginalized models outputs.
class transformers.models.rag.modeling_rag. RetrievAugLMOutput < source >
( logits : typing.Optional[torch.FloatTensor] = None doc_scores : typing.Optional[torch.FloatTensor] = None past_key_values : transformers.cache_utils.Cache | None = None retrieved_doc_embeds : typing.Optional[torch.FloatTensor] = None retrieved_doc_ids : typing.Optional[torch.LongTensor] = None context_input_ids : typing.Optional[torch.LongTensor] = None context_attention_mask : typing.Optional[torch.LongTensor] = None question_encoder_last_hidden_state : typing.Optional[torch.FloatTensor] = None question_enc_hidden_states : tuple[torch.Fl
