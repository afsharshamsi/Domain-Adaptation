import model
from model import *
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-base")
# suppose we load BART

from bigmodelvis import Visualization
print("before modify")
Visualization(model).structure_graph()
trainable = count_trainable_parameters(model)
print("Trainable parameters of LoRa model: ", trainable)
print()
"""
The white part is the name of the module.
The green part is the module's type.
The blue part is the tunable parameters, i.e., the parameters that require grad computation.
The grey part is the frozen parameters, i.e., the parameters that do not require grad computation.
The red part is the structure that is repeated and thus folded.
The purple part is the delta parameters inserted into the backbone model.
"""

from opendelta import LoraModel
delta_model = LoraModel(backbone_model=model, modified_modules=['fc2'])
print("after modify")
delta_model.log()

trainable_params, delta_params, total_params  = count_trainable_parameters(delta_model.backbone_model, delta_model.modified_modules)
print(f"Trainable Params: {trainable_params} | Delta Params: {delta_params} | Total Params: {total_params}")
print()

# This will visualize the backbone after modification and other information.

delta_model.freeze_module(exclude=["deltas", "layernorm_embedding"], set_state_dict=True)
print("after freeze")
delta_model.log()
trainable_params, delta_params, total_params  = count_trainable_parameters(delta_model.backbone_model, delta_model.modified_modules)
print(f"Trainable Params: {trainable_params} | Delta Params: {delta_params} | Total Params: {total_params}")
print()
# The set_state_dict=True will tell the method to change the state_dict of the backbone_model to maintaining only the trainable parts.