{
  description = "A very basic flake development environment";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/f3dab3509afca932f3f4fd0908957709bb1c1f57";

  outputs = { self, nixpkgs }:
    let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
    in
    {
      # nix fmt
      formatter.x86_64-linux = pkgs.nixpkgs-fmt;
      # nix build
      packages.x86_64-linux.default = pkgs.hello;
      # nix develop
      devShells.x86_64-linux.default = pkgs.mkShell {
        #packages = with pkgs; [ hello fish ];
        packages = builtins.attrValues {
          inherit (pkgs) gcc11 mpi;
          # mpirun (Open MPI) 4.1.4
          # g++ (GCC) 11.3.0
        };

        shellHook = ''
          # fix terminal when using alacritty
          export TERM=xterm-256color
          exec fish
        '';
      };
    };
}
