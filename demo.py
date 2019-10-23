

from model import create_model, load_model
import torch

if __name__ == '__main__':
    num_classes = 80
    head_conv = 256
    heads = {'hm': num_classes,
             'wh': 2 ,
             'reg': 2}

    load_model_path = 'ctdet_coco_dla_2x.pth'
    save_script_pt = 'centernet.pt'
    device = 0

    model = create_model('dla_34', heads, head_conv)
    model = load_model(model, load_model_path)
    model = model.to(device)
    model.eval()

    input_var = torch.zeros([1, 3, 512, 512], dtype=torch.float32).cuda()

    traced_script_module = torch.jit.trace(model, input_var)
    traced_script_module.save(save_script_pt)
    traced_script_module = torch.jit.load(save_script_pt)
