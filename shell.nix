with import <nixpkgs> {};
stdenv.mkDerivation rec {
  name = "dev";
  env = buildEnv {
    name = name;
    paths = buildInputs;
  };
  buildInputs = [
    python36
    python36Packages.six
  ];
}
