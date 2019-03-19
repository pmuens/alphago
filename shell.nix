with import <nixpkgs> {};
stdenv.mkDerivation rec {
  name = "dev";
  env = buildEnv {
    name = name;
    paths = buildInputs;
  };
  buildInputs = [
    gnugo
    python36
    python36Packages.six
    python36Packages.numpy
    python36Packages.matplotlib
    python36Packages.Keras
    python36Packages.tensorflow
    python36Packages.h5py
    python36Packages.flask
  ];
}
