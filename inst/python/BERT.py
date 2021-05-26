from transformers import *
import torch
# Transformers has a unified API
# for 10 transformer architectures and 30 pretrained weights.
#            Architecture      | Model          | Tokenizer          
MODELS = {    'BERT':          (BertModel,       BertTokenizer),
              'GPT':           (OpenAIGPTModel,  OpenAIGPTTokenizerFast),
              'GPT-2':         (GPT2Model,       GPT2TokenizerFast),
              'CTRL':          (CTRLModel,       CTRLTokenizer),
              'Transformer-XL':(TransfoXLModel,  TransfoXLTokenizer),
              'XLNet':         (XLNetModel,      XLNetTokenizerFast),
              'XLM':           (XLMModel,        XLMTokenizer),
              'DistilBERT':    (DistilBertModel, DistilBertTokenizerFast),
              'RoBERTa':       (RobertaModel,    RobertaTokenizerFast),
              'XLM-RoBERTa':   (XLMRobertaModel, XLMRobertaTokenizerFast),
              'GPT-2-LMHead':  (GPT2LMHeadModel, GPT2TokenizerFast),
              'CamenBERT':     (CamembertModel,  CamembertTokenizerFast),
              'FlauBERT':      (FlaubertModel,   FlaubertTokenizer),
              'ALBERT':        (AlbertModel,     AlbertTokenizerFast),
              'T5':            (T5Model,         T5TokenizerFast),
              'BART':          (BartModel,       BartTokenizerFast),
              'ELECTRA':       (ElectraModel,    ElectraTokenizerFast),
              'Reformer':      (ReformerModel,   ReformerTokenizerFast),
              'MarianMT':      (MarianMTModel,   MarianTokenizer),
              'Longformer':    (LongformerModel, LongformerTokenizerFast)
         }

class Embedder():
	def __init__(self, path = None, architecture = "BERT", use_cuda = False):
		model_class, tokenizer_class = MODELS[architecture]
		use_cuda = use_cuda & torch.cuda.is_available()
		self.device = "cuda:0" if use_cuda else "cpu"
		self.device_id = 0 if use_cuda else -1
		self.model = model_class.from_pretrained(path).to(self.device)
		self.tokenizer = tokenizer_class.from_pretrained(path)
		self.nlp_feature_extraction = pipeline("feature-extraction", model = self.model, tokenizer = self.tokenizer, device = self.device_id)
	def tokenize(self, text):
		output = self.tokenizer.tokenize(text)
		return(output)
	def embed_tokens(self, text):
		output = self.nlp_feature_extraction(text)
		return(output)
	def generate(self, text, max_length = 50, **kwargs):
		input_ids = self.tokenizer.encode(text, return_tensors = "pt").to(self.device)
		generated = self.model.generate(input_ids, kwargs, max_length = max_length, truncation = True)
		newtext = self.tokenizer.decode(generated.tolist()[0])
		return(newtext)
	def embed_sentence(self, text, max_length = 512, **kwargs):
		input_ids = self.tokenizer.encode(text, add_special_tokens = True,
										  max_length = max_length, truncation = True,
										  return_tensors = 'pt').to(self.device)		
		with torch.no_grad():
			output_tuple = self.model(input_ids, **kwargs)
  		
		output = output_tuple[0].squeeze()
		output = output.mean(dim = 0)
		output = output.cpu().numpy()
		return(output)
		
def download(model_name = "bert-base-multilingual-uncased", path = None, architecture = "BERT"):
	model_class, tokenizer_class = MODELS[architecture]
	# load pretrained model/tokenizer
	tokenizer = tokenizer_class.from_pretrained(pretrained_model_name_or_path = model_name)
	model = model_class.from_pretrained(pretrained_model_name_or_path = model_name, output_hidden_states = True, output_attentions = False)
	# save them to disk
	tokenizer.save_pretrained(path)
	model.save_pretrained(path)
	return(path)
