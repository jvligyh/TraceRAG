# Hugging Face Hub: Using Sentence Transformers

Source: https://huggingface.co/docs/hub/sentence-transformers
Scope: official-documentation-excerpt; not a full-paper archive.

Hub documentation
Using Sentence Transformers at Hugging Face
Hub 🏡 View all docs AWS Trainium & Inferentia Accelerate Argilla AutoTrain Bitsandbytes CLI Chat UI Dataset viewer Datasets Deploying on AWS Diffusers Distilabel Evaluate Google Cloud Google TPUs Gradio Hub Hub Python Library Huggingface.js Inference Endpoints (dedicated) Inference Providers Kernels LeRobot Leaderboards Lighteval Microsoft Azure OpenEnv Optimum PEFT Reachy Mini Safetensors Sentence Transformers TRL Tasks Text Embeddings Inference Text Generation Inference Tokenizers Trackio Transformers Transformers.js Xet smolagents timm Search documentation EN JA KO PT ZH API docs Join the Hugging Face community
and get access to the augmented documentation experience Collaborate on models, datasets and Spaces Faster examples with accelerated inference Switch between documentation themes Sign Up
to get started
Copy page
Using Sentence Transformers at Hugging Face
sentence-transformers is a library that provides easy methods to compute embeddings (dense vector representations) for sentences, paragraphs and images. Texts are embedded in a vector space such that similar text is close, which enables applications such as semantic search, clustering, and retrieval.
Exploring sentence-transformers in the Hub
You can find over 500 hundred sentence-transformer models by filtering at the left of the models page . Most of these models support different tasks, such as doing feature-extraction to generate the embedding, and sentence-similarity as a way to determine how similar is a given sentence to other. You can also find an overview of the official pre-trained models in the official docs .
All models on the Hub come up with features:
An automatically generated model card with a description, example code snippets, architecture overview, and more.
Metadata tags that help for discoverability and contain information such as license.
An interactive widget you can use to play out with the model directly in the browser.
An Inference Providers widget that allows to make inference requests.
Using existing models
The pre-trained models on the Hub can be loaded with a single line of code Copied
from sentence_transformers import SentenceTransformer
model = SentenceTransformer( 'model_name' )
Here is an example that encodes sentences and then computes the distance between them for doing semantic search. Copied
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer( 'multi-qa-MiniLM-L6-cos-v1' )
query_embedding = model.encode( 'How big is London' )
passage_embedding = model.encode([ 'London has 9,787,426 inhabitants at the 2011 census' ,
'London is known for its financial district' ])
print ( "Similarity:" , util.dot_score(query_embedding, passage_embedding))
If you want to see how to load a specific model, you can click Use in sentence-transformers and you will be given a working snippet that you can load it!
Sharing your models
You can share your Sentence Transformers by using the save_to_hub method from a trained model. Copied
from sentence_transformers import SentenceTransformer
# Load or train a model
model.save_to_hub( "my_new_model" )
This command creates a repository with an automatically generated model card, an inference widget, example code snippets, and more! Here is an example.
Additional resources
Sentence Transformers library .
Sentence Transformers docs .
Integration with Hub announcement . Update on GitHub
← Sample Factory SetFit →
