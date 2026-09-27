# Changelog

All notable changes to POOCH will be documented in this file.

The project follows [Semantic Versioning](https://semver.org/).

## [0.1.1] - 2026-09-27

### Added
- add slurm
- pkgs for chain file generation added to yml file.

### Fixed
- enables slurm
- chain files solely also are acceptable
- make symlink of methylation when MET_HAP_ON_REF_FILEs are given
- generate chromAssigns.filtered.txt using MET_HAP2_ON_REF_FILE before launch nextflow pipeline
- output file is bed.gz
- change modkit version to 0.6.4
- add rustybam
- uscs tools added
- tmp folder reset
- chmod 777
- add java
- remove crossmap from tool check
- add --chromid a and change from crossmap to CrossMap
- detect Mm tag
- fix bug in pooch-predict

## [0.1.0] - 2026-09-25

### Fixed
- update release workflow permissions

## 0.1.0

### Added

- Initial public release of POOCH.
- Parent-of-origin classification of haplotypes.
- Chromosome assignment of near-complete genome assemblies.
- Support for ONT and PacBio HiFi sequencing data.
- Support for personalized diploid genome or split haplotype FASTA input.
- Support for FASTQ, BAM, and methylation BED input modes.
- Methylation liftover to reference genome coordinates.
- Parent-of-origin prediction per chromosome.
