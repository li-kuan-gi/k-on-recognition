import torch


def convert_to_onnx(model, path, input_names, output_names):
    model.to('cpu')
    model.train(False)
    dummy_input = torch.randn(
        1, 3, 224, 224, requires_grad=True, device=torch.device('cpu'))
    torch.onnx.export(model, dummy_input, path, export_params=True,
                      input_names=input_names, output_names=output_names,
                      verbose=True, opset_version=11)
