# raw-ngp

A ngp-based approach for training a NeRF network on raw image data from a light stage.

This repository is based on [nerf_template](https://github.com/ashawkey/nerf_template), which is a reduced version of [torch-ngp](https://github.com/ashawkey/torch-ngp).

Supported Features include:
* light directional conditioning for relighting
* pose refinement like in BARF and BAANGP
* HDR network predictions and exposure merging
* postprocessing options for the model's predictions
* learning from masked light stage images
* light stage datasets are uploaded soon...

![poster-5](https://github.com/je-lehmann/ngp_raw_nerf/assets/49212649/6684c5a8-4d56-4ce7-8568-0ea03a2dfbd9)

Overview of the complete reconstruction pipeline: Image acquisition is executed on a light stage in a programmable manner. Automated image masking, enabled by a Segment Anything Model (SAM) is applied to mask out irrelevant scene information. The NGP-like MLP is conditioned on the efficiently sampled positional input x, a refined view direction d and single light direction l to output the color estimate c and density estimate sigma. Optimization is conducted directly on raw image data, allowing for extended postprocessing of the linear predictions. Novel view and light synthesis can be achieved.




https://github.com/je-lehmann/ngp_raw_nerf/assets/49212649/b2d19047-cdca-42c3-a166-d3693b7b6a25



# Install

### Install with pip
```bash
pip install -r requirements.txt
```

### Build extension (optional)
By default, use [`load`](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load) to build the extension at runtime.
However, this may be inconvenient sometimes.
Therefore, `setup.py` can be used to build each extension:
```bash
# install all extension modules
bash scripts/install_ext.sh

# if you want to install manually, here is an example:
cd raymarching
python setup.py build_ext --inplace # build ext only, do not install (only can be used in the parent directory)
pip install . # install to python path (you still need the raymarching/ folder, since this only install the built extension.)
```

### Acknowledgement
This repository is based on:
* [nerf_template](https://github.com/ashawkey/nerf_template)
* [torch-ngp](https://github.com/ashawkey/torch-ngp)
* [MultiNeRF](https://github.com/google-research/multinerf)
* [nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
* [nerfacc](https://github.com/KAIR-BAIR/nerfacc)
* [BARF](https://github.com/chenhsuanlin/bundle-adjusting-NeRF)
* [BAA-NGP](https://github.com/IntelLabs/baa-ngp)
