# Testing your docker image

## Requirements

 * **docker** 
 
 `apt` installation: https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository

Remember to follow pos-installation steps: https://docs.docker.com/engine/install/linux-postinstall/
 
 * **nvidia-container-toolkit**

 In order to share the GPU between the host and the docker container you need to install `nvidia-docker-toolkit`

 `apt` installation: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt

 Configure docker to use nvidia runtime: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker



> **Warning**
>
>For nvidia-container-toolkit versions older than v1.12.0 user would have to manually mount vulkan icd.d JSON file into the container (https://github.com/NVIDIA/nvidia-container-toolkit/issues/16)
>
>For a fictional user "me" the Driver Manifest search path might look like the following:
>- /home/me/.config/vulkan/icd.d
>- /etc/xdg/vulkan/icd.d
>- /usr/local/etc/vulkan/icd.d
>- /etc/vulkan/icd.d
>- /home/me/.local/share/vulkan/icd.d
>- /usr/local/share/vulkan/icd.d
>- /usr/share/vulkan/icd.d
>
> Check your `nvidia-container-toolkit` with the following command `nvidia-ctk --version`

