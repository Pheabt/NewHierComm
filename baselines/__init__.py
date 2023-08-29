from .tiecomm import TieCommAgent
from .hiercomm import HierCommAgent
from .commnet import CommNetAgent
from .tarmac import TarCommAgent
from .magic import MAGICAgent
# from .gacomm import GACommAgent
from .models import MLP, Attention,GNN, Attention_Noise


REGISTRY = {}
REGISTRY["tiecomm"] = TieCommAgent
REGISTRY["tiecomm_random"] = TieCommAgent
REGISTRY["tiecomm_one"] = TieCommAgent
REGISTRY["tiecomm_default"] = TieCommAgent
REGISTRY["tiecomm_wo_inter"] = TieCommAgent
REGISTRY["tiecomm_wo_intra"] = TieCommAgent

REGISTRY["hiercomm"] = HierCommAgent




REGISTRY["commnet"] = CommNetAgent
REGISTRY["ic3net"] = CommNetAgent
REGISTRY["tarmac"] = TarCommAgent
REGISTRY["magic"] = MAGICAgent

REGISTRY["ac_mlp"] = MLP
REGISTRY["ac_att"] = Attention
REGISTRY["ac_att_noise"] = Attention_Noise

REGISTRY["gnn"] = GNN


# REGISTRY["gacomm"] = GACommAgent