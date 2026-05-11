import sys, types, torch, warnings
warnings.filterwarnings('ignore')
import dreams.utils.data as du; import dreams.utils.dformats as dformats
import dreams.utils.spectra as su; import dreams.models.dreams.dreams as dm
import dreams.models.dreams.layers as dl
import dreams.models.layers.fourier_features as ff
import dreams.models.layers.feed_forward as fw
for ns in ['msml','msml.models','msml.models.dreams','msml.models.layers','msml.utils']:
    sys.modules[ns] = types.ModuleType(ns)
sys.modules['msml.models.dreams.dreams'] = dm
sys.modules['msml.models.dreams.layers'] = dl
sys.modules['msml.models.layers.fourier_features'] = ff
sys.modules['msml.models.layers.feed_forward'] = fw
sys.modules['msml.utils.data'] = du; sys.modules['msml.utils.dformats'] = dformats
sys.modules['msml.utils.spectra'] = su

kw = {'map_location':'cpu', 'weights_only':False}
s = torch.load('dreams/models/pretrained/ssl_model.ckpt', **kw)
e = torch.load('dreams/models/pretrained/embedding_model.ckpt', **kw)
sd_s = s['state_dict']; sd_e = e['state_dict']

print('=== ssl_model.ckpt ===')
has_head = [k for k in sd_s if 'ff_out' in k or 'ro_out' in k]
print(f'  key数: {len(sd_s)}, 含任务头: {has_head}')

print()
print('=== embedding_model.ckpt ===')
bp = [k for k in sd_e if k.startswith('backbone.')]
eh = [k for k in sd_e if 'head' in k]
print(f'  key数: {len(sd_e)}')
print(f'  backbone. 前缀: {len(bp)}/{len(sd_e)} 个key')
print(f'  含head层: {eh}')
print(f'  含ff_out: {any("ff_out" in k for k in sd_e)}')
print(f'  含ro_out: {any("ro_out" in k for k in sd_e)}')

# Compare backbones (strip prefix from embedding_model)
bk_s = [k for k in sd_s if not any(x in k for x in ['ff_out','ro_out','mz_masking'])]
bk_e_set = set(k.replace('backbone.','') for k in sd_e if k.startswith('backbone.') and 'head' not in k)
bk_s_set = set(bk_s)
print()
print(f'ssl backbone key数: {len(bk_s_set)}')
print(f'emb backbone key数(去前缀): {len(bk_e_set)}')
print(f'共同backbone key: {len(bk_s_set & bk_e_set)}')
print(f'ssl独有: {sorted(bk_s_set - bk_e_set) or "无"}')
print(f'emb独有: {sorted(bk_e_set - bk_s_set) or "无"}')
