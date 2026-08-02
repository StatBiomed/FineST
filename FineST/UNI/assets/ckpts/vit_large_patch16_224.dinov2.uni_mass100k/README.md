Put `pytorch_model.bin` here.

## 2026.04.16 
LLY Copy 'pytorch_model.bin' from 
- /home/lingyu/ssd2/.cache/huggingface/hub/models--MahmoodLab--uni/snapshots/b55a5ec6cade1a39edfe6534189a9b8ca7a022f0
- pytorch_model.bin


    >>> from huggingface_hub import login
    >>> login() 

        _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
        _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
        _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
        _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
        _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

    Enter your token (input will not be visible): 
    Add token as git credential? (Y/n) n



    >>> import timm
    >>> from timm.data import resolve_data_config
    >>> from timm.data.transforms_factory import create_transform
    >>> model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
    config.json: 100%|███████████████████████████████████████████████████████████████████| 686/686 [00:00<00:00, 88.9kB/s]
    pytorch_model.bin: 100%|██████████████████████████████████████████████████████████| 1.21G/1.21G [00:11<00:00, 101MB/s]



    >>> from huggingface_hub import hf_hub_download
    >>> path = hf_hub_download(repo_id="MahmoodLab/uni", filename="pytorch_model.bin")
    >>> print(path)
    /ssd2/users/lingyu/.cache/huggingface/hub/models--MahmoodLab--uni/snapshots/b55a5ec6cade1a39edfe6534189a9b8ca7a022f0/pytorch_model.bin

