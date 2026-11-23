# Archived SOTA Attacks (Baruch et al. ALIE, Xie et al. IPM, Fang et al. Min-Max)
# ==============================================================================
#
# def alie_attack(global_state, trained_state, z_max=1.0):
#     out = {}
#     for k in global_state:
#         g = global_state[k].float()
#         t = trained_state[k].float()
#         delta = t - g
#         std = delta.std().clamp(min=1e-5)
#         out[k] = (g - z_max * std * torch.sign(delta)).cpu()
#     return out
#
# def ipm_attack(global_state, trained_state, epsilon=0.5):
#     out = {}
#     for k in global_state:
#         g = global_state[k].float()
#         t = trained_state[k].float()
#         delta = t - g
#         out[k] = (g - epsilon * delta).cpu()
#     return out
#
# def min_max_attack(global_state, trained_state):
#     out = {}
#     for k in global_state:
#         g = global_state[k].float()
#         t = trained_state[k].float()
#         delta = t - g
#         norm = delta.norm().clamp(min=1e-5)
#         unit_reverse = -delta / norm
#         out[k] = (g + unit_reverse * (norm * 0.9)).cpu()
#     return out
