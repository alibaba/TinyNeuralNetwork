import torch
import torchvision

from tinynn.graph.tracer import model_tracer, trace

with model_tracer():
    # Prapare the model
    # It's okay to put the construction of the model out of the
    # with-block, but actually leave it here would be better.
    # The latter one guarantees that the arguments that is used
    # to build the model is caught, while the other one doesn't.
    model = torchvision.models.alexnet()
    model.eval()

    # Provide a viable input for the model
    dummy_input = torch.rand((1, 3, 224, 224))

    # After tracing the model, we will get a TraceGraph object
    graph = trace(model, dummy_input)

    # (Optional) We can modify some of the properties here
    # even if it is used to build the model
    model.avgpool.output_size = (7, 7)

    # We can use it to generate the code for the original model
    # But be careful that it has to be in the with-block.
    graph.generate_code('my_alexnet.py', 'my_alexnet.pth', 'Alexnet')
