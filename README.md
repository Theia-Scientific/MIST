# tstiler: A command line utility and Python package for running tiled inference

[![CI](https://github.com/Theia-Scientific/tstiler/actions/workflows/ci.yml/badge.svg)](https://github.com/Theia-Scientific/tstiler/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Theia-Scientific/tstiler/graph/badge.svg?token=Fy1sOhp76u)](https://codecov.io/gh/Theia-Scientific/tstiler)

A Command Line Interface (CLI) application for running a tiled inference that
works with bounding boxes, instance segmentations, and merges large objects
spanning multiple tiles into single instances. Inspired by the [SAHI],
[Patched], and [dask_relabeling] tiling inference packages.

The SAHI tiler only does bounding boxes and does not merge large instances. The
Patched tiler does instance segmentations but does not merge large instances.
The `dask_relabeling` package does instance segmentations and merging large
instances, but it does not work with YOLO models and GPU-powered inference.

1. [Prerequisites](#prerequisites)
   1. [Python](#prerequisites-python)
      1. [Ubuntu](#prerequisites-python-ubuntu)
      2. [macOS](#prerequisites-python-macos)
   2. [pipx](#prerequisites-pipx)
      1. [Ubuntu](#prerequisites-pipx-ubuntu)
      2. [macOS](#prerequisites-pipx-macos)
   3. [Command Line Utilities](#prerequisites-cli-apps)
      1. [Ubuntu](#prerequisites-cli-apps-ubuntu)
      2. [macOS](#prerequisites-cli-apps-macos)
2. [Installation](#installation)
   1. [pipx](#installation-pipx) (recommended)
   2. [Source](#installation-source)
3. [Upgrade](#upgrade)
   1. [pipx](#upgrade-pipx)
   2. [Source](#upgrade-source)
4. [Usage](#usage)
   1. [Terminal](#usage-terminal)
5. [Contributing](#contributing)
6. [License](#license)

## Prerequisites

All of the prerequisites may already be installed and configured by the
superuser, a.k.a. root, of the computer. The prerequisites only need to be
installed and configured once per machine.

### Python

<a name="prerequisites-python"></a>

The [Python] programming language is needed to run the `tstiler` Command Line
Interface (CLI) application and/or use the `tstiler` package in other Python
scripts or [Jupyter] notebooks. Both macOS and Ubuntu Linux have the Python
programming language installed, but it is generally reserved for the operating
system (OS) to use and is an older version. It is best practice to install a
newer version that is separate from the system-provided Python version.

#### Ubuntu

<a name="prerequisites-python-ubuntu"></a>

1. Add the "[deadsnakes]" Ubuntu Personal Package Archives (PPA).

    ```sh
    sudo add-apt-repository ppa:deadsnakes/ppa
    ```
    
2. Obtain the latest packages from the PPA.

   ```sh
   sudo update
   ```
   
3. Install Python v3.11 or newer.

   ```sh
   sudo apt install python3.11
   ```
   
4. Install the `venv` package.

   ```sh
   sudo apt install python3.11-venv
   ```
   
#### macOS

<a name="prerequisites-python-macos"></a>

1. Install [Homebrew] if it is not already installed.

   ```sh
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
   
2. Install Python v3.11 or newer.

   ```sh
   brew install python@3.11
   ```

### pipx

<a name="prerequisites-pipx"></a>

The [pipx] utility enables distribution of Python-based CLI applications, like
`tstiler`, to be installed for all users with all of the appropriate dependencies
within an isolated environment. It is the recommended installation for the
`tstiler` application.

#### Ubuntu

<a name="prerequisites-pipx-ubuntu"></a>

1. Create a virtual environment for `pipx` and Python v3.11 or newer.

   ```sh
   sudo python3.11 -m venv --upgrade-deps /opt/pipx
   ```
   
2. Install `pipx` for all users.

   ```sh
   sudo /opt/pipx/bin/pip install pipx
   ```
   
3. Ensure the `pipx` command is available to all users.

   ```sh
   sudo ln -s /opt/pipx/bin/pipx /usr/local/bin/pipx
   ```

4. Add `pipx` to the `PATH` environment variable.

   ```sh
   pipx ensurepath
   ```
   
5. Add `pipx` for all users.

   ```sh
   sudo pipx ensurepath --global
   ```
   
Post-installation, the `pipx` application can be upgraded with the following
command:

```sh
sudo /opt/pipx/bin/pip install --upgrade pipx
```

#### macOS

<a name="prerequisites-pipx-macos"></a>

1. Install `pipx` using [Homebrew].

   ```sh
   brew install pipx
   ```
   
2. Add `pipx` to the `PATH` environment variable.

   ```sh
   pipx ensurepath
   ```
   
3. Add `pipx` for all users.

   ```sh
   sudo pipx ensurepath --global
   ```
   
Post-installation, the `pipx` application can be upgraded with the following
command:

``` sh
brew update && brew upgrade pipx
```

### Command Line Interface Applications

<a name="prerequisites-cli-apps"></a>

#### Ubuntu

<a name="prerequisites-cli-apps-ubuntu"></a>

1. Install applications from the public repositories.

   ```sh
   sudo apt install imagemagick jq tree
   ```
   
2. Get the latest version of `yq` from source.

   ```sh
   sudo wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq && sudo chmod +x /usr/bin/yq
   ```
  
#### macOS

<a name="prerequisites-cli-apps-macos"></a>

1. Install applications using [homebrew].

   ```sh
   brew install imagemagick jq tree yq
   ```
   
## Installation

### pipx (recommended)

<a name="installation-pipx"></a>

1. Download the ZIP archive of the source code. A link can be obtained from
   Theia Scientific personnel.
2. Ensure `pipx` is installed. See the [Prerequisites](#prerequisites).

   ```sh
   $ pipx --version
   1.7.1
   ```
   
3. Install `tstiler` command globally for all users.

   ```sh
   sudo pipx install --global --python python3.11 "./tstiler-0.1.0.zip"
   ```
   
4. Verify `tstiler` command is available.

   ```sh
   $ tstiler --version
   tstiler 0.1.0
   ```
   
### Source

<a name="installation-source"></a>

1. Clone this repository.

   ```sh
   git clone https://github.com/Theia-Scientific/tstiler.git && cd tstiler
   ```

2. Create a virtual environment.

   ```sh
   python3 -m venv .venv
   ```

3. Activate the virtual environment.

   ```sh
   source .venv/bin/activate
   ```
   
   or if [direnv] is installed:
   
   ```sh
   cp .envrc.example .envrc
   ```
   
   followed by:
   
   ```sh
   direnv allow
   ```

4. Upgrade `pip` to the latest version.

   ```sh
   python3 -m pip install --upgrade pip
   ```

5. Locally install the package, utility, and its dependencies. This will create
   the `tstiler` command within the virtual environment. This also installs the
   Slack feature for sending notifications to a Slack channel.

   ```sh
   python3 -m pip install -e .
   ```

## Upgrade

### pipx (recommended)

<a name="upgrade-pipx"></a>

1. Download the ZIP archive of the source code. A link can be obtained from
   Theia Scientific personnel.
2. Upgrade the `tstiler` application via `pipx`.

   ```sh
   sudo pipx install --global --python python3.11 --force "./tstiler-0.1.0.zip"
   ```
   
3. Verify new version.

   ```sh
   $ tstiler --version
   tstiler 0.1.0
   ```
   
### Source

<a name="upgrade-source"></a>

1. Navigate to the root of the source tree.

   ```sh
   cd ~/Code/tstiler
   ```

2. Activate the virtual environment.

   ```sh
   source .venv/bin/activate
   ```
   
   or if [direnv] is installed, the virtual environment will automatically be
   activated.
   
3. Pull the latest changes on `main`.

   ```sh
   git pull
   ```
   
4. Upgrade the `tstiler` application within the virtual environment.

   ```sh
   python -m pip install --upgrade -e .
   ```

5. Verify new version.

   ```sh
   $ tstiler --version
   tstiler 0.1.0
   ```

## Usage

### Terminal

<a name="usage-terminal"></a>

TODO: Add steps

## Contributing

1. Clone this repository.

   ```sh
   git clone https://github.com/Theia-Scientific/tstiler.git && cd tstiler
   ```

2. Create a virtual environment.

   ```sh
   python3 -m venv .venv
   ```

3. Activate the virtual environment.

   ```sh
   source .venv/bin/activate
   ```

   or if [direnv] is installed:
   
   ```sh
   cp .envrc.example .envrc
   ```
   
   followed by:
   
   ```sh
   direnv allow
   ```

4. Install the package, utility, the required dependencies, and the development
   dependencies.

   ```sh
   python3 -m pip install -e ".[dev]"
   ```

5. Create a local branch.

   ```sh
   git checkout -b feature-awesome-new-feature
   ```

6. Modify the code.
7. Run the tests.

   ```sh
   pytest --color=yes
   ```

8. Commit changes to your local branch.

   ```sh
   git add -A && git commit -m "Add new feature"
   ```

9. Push your local branch to GitHub to create a Pull Request (PR).

   ```sh
   git push origin feature-awesome-new-feature
   ```

10. Create a Pull Request (PR) in GitHub.
11. Wait for CI to complete.
12. Add comment to PR that it is ready to review.

## License

Copyright (C) 2025 Theia Scientific, LLC. All rights reserved.

[dask_relabeling]: https://github.com/TheJacksonLaboratory/dask_relabeling
[jupyter]: https://jupyter.org/
[patched]: https://github.com/Koldim2001/YOLO-Patch-Based-Inference
[sahi]: https://github.com/obss/sahi
