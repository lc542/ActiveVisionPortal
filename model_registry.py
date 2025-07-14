from models.IRL import entry as irl_entry
from models.Gazeformer import entry as gazeformer_entry
from models.HAT import entry as hat_entry
from models.CLIPGaze import entry as clipgaze_entry
from models.Scanpaths import entry as scanpaths_entry

MODEL_REGISTRY = {
    'irl': irl_entry,
    'gazeformer': gazeformer_entry,
    'hat': hat_entry,
    'clipgaze': clipgaze_entry,
    'scanpaths': scanpaths_entry,
}
