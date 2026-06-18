# Guide to /testing_data directory

All the data stored here is used to test the expected functionality of the source code

## /AtomicDescriptors

Data generated from /ReadORCA6Outputs, the `.mol` files contain atomic SOAP descriptors, with their respective electronic energies and gradients, ready to train an MLP.

## /InitialTests

Data generated here is used to test the core expected functionality of the `Molecule` class.

## /ReadORCA6Outputs

Data Generated here is used to test the reading and writing or ORCA6 files, part of this software is to be the middle man between quantum chemistry software and workflows to process quantum chemical data

## /TS

Databank of all the different TS found in the literature, used for testing `Molecule` object to SMARTS strings