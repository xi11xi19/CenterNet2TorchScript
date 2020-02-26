# DCN C PLUS CPLUS PLUGIN

## usage
void handle = dlopen("libdcn_v2_cuda_forward_v2.so", RTLD_LAZY);

int gpu_id = 0;
torch::jit::script::Module module = 
torch::jit::load("centernet.pt", torch::Device(torch::DeviceType::CUDA, gpu_id));
