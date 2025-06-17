from models.IRL import entry as irl_entry
from models.Gazeformer import entry as gazeformer_entry

MODEL_REGISTRY = {
    'irl': irl_entry,
    'gazeformer': gazeformer_entry,
}
