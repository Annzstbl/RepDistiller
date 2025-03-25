import torch



# detr_checkpoint_file = '/data3/litianhao/hsmot/motip/r50_deformable_detr_coco.pth'
detr_checkpoint_file = '/data3/litianhao/hsmot/motr/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint_8ch_interpolate.pth'
distill_file = '/data3/litianhao/hsmot/distill/hintlayer:2_3_4_hintweights:1_1_1_firstconvLrScale:10_1/resnet8_last.pth'
save_file = '/data3/litianhao/hsmot/motr/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint_8ch_interpolate_distill.pth'

# load two models and replace backbone of detr with distill_file

detr = torch.load(detr_checkpoint_file)
distill = torch.load(distill_file)

pass

detr_model = detr['model']
distill_model = distill['model']

match_keys = []
unmatch_keys = []

for n, p in distill_model.items():
    detr_name = f'backbone.0.{n}'
    if detr_name in detr_model and detr_model[detr_name].shape == p.shape:
        match_keys.append(n)
        detr_model[detr_name] = p
    else:
        unmatch_keys.append(n)

print(f'Total distill model keys: {len(distill_model)}')
print(f'Matched keys: {len(match_keys)}')
print(f'Unmatched keys: {len(unmatch_keys)}')
print(f'Unmatched keys: {unmatch_keys}')

detr['model'] = detr_model
torch.save(detr, save_file)
print(f'Save distill model to {save_file}')
pass