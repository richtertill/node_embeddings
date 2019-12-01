# import embedding methods
from embedding.bernoulli import Bernoulli
from embedding.KL import KL

def create_model_from_dict(m_dict):
	if m_dict['name'] == "Bernoulli":
		model = Bernoulli(m_dict["emb_dim"], m_dict['distance_meassure'])
	elif m_dict['name'] == "KL":
		model = KL(m_dict["emb_dim"])
	else:
		print("Error in model dict.")

	return model
	