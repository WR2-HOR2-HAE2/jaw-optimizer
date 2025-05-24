{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    cargo
    rustc
    rust-analyzer
    clippy
    rustfmt
    dav1d
    pkg-config
  ];
}
