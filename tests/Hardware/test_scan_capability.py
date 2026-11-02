"""PERF-02: the driver read matrix must stay honest about the code.

Every claim here is checked against the driver sources, not against the
matrix itself: a driver may only be described as combining channels when
its reader really issues one command for both, and an unverified read may
only ever be reachable through an explicit opt-in.
"""

from __future__ import annotations

import pytest

from NanoVNASaver.Hardware.Capability import SweepCapability
from NanoVNASaver.Hardware.LiteVNA64 import LiteVNA64
from NanoVNASaver.Hardware.NanoVNA import NanoVNA
from NanoVNASaver.Hardware.NanoVNA_V2 import NanoVNA_V2
from NanoVNASaver.Hardware.ScanCapability import (
    BINARY_OPT_IN,
    BINARY_UNSUPPORTED,
    DRIVERS,
    GENERIC,
    PATH_FIFO,
    PATH_SCAN,
    PATH_SCAN_MASK,
    PATH_SWEEP,
    PATHS,
    ScanCapability,
    ScanPreferences,
    for_driver,
    hardware_unverified_drivers,
    scan_capability_for,
)

OPT_IN = ScanPreferences(combined_text=True, binary_scan=True)


class TestMatrixShape:
    @staticmethod
    @pytest.mark.parametrize("name", sorted(DRIVERS))
    def test_every_entry_names_itself_and_a_known_path(name):
        entry = DRIVERS[name]
        assert entry.driver == name
        assert entry.path in PATHS
        assert entry.evidence, "a claim without a citation is a guess"

    @staticmethod
    def test_the_matrix_is_immutable():
        with pytest.raises(TypeError):
            DRIVERS["NanoVNA"] = GENERIC  # type: ignore[index]

    @staticmethod
    def test_an_unknown_driver_gets_the_generic_entry():
        assert for_driver("Not A Device") == GENERIC
        assert GENERIC.path == PATH_SWEEP

    @staticmethod
    def test_the_generic_entry_offers_no_optimisation():
        assert GENERIC.requested(OPT_IN) == PATH_SWEEP


class TestClaimsMatchTheCode:
    @staticmethod
    def test_only_the_v2_family_claims_a_fifo_read():
        fifo = {
            name for name, entry in DRIVERS.items() if entry.path == PATH_FIFO
        }
        assert fifo == {"NanoVNA_V2", "LiteVNA64"}
        # ...and both really do speak binary records.
        for name in fifo:
            assert DRIVERS[name].binary_transport
            assert DRIVERS[name].combines_channels

    @staticmethod
    def test_only_the_nanovna_text_family_claims_a_masked_scan():
        masked = {
            name
            for name, entry in DRIVERS.items()
            if entry.path == PATH_SCAN_MASK or entry.scan_mask
        }
        # No row claims the mask today: the only driver that emits it picks
        # it at runtime from the firmware, which ``with_runtime`` folds in.
        assert masked == set()
        assert DRIVERS["NanoVNA"].path == PATH_SCAN

    @staticmethod
    def test_a_one_port_driver_never_claims_to_combine_channels():
        # TinySA/TinySA_Ultra read a single channel: combining is meaningless
        # and must not be claimed, or a caller would ask for an S21 that the
        # device cannot produce.
        for name in ("TinySA", "TinySA_Ultra", "AVNA"):
            assert not DRIVERS[name].combines_channels

    @staticmethod
    def test_the_binary_state_is_never_verified_without_evidence():
        for name, entry in DRIVERS.items():
            assert entry.binary_scan in {BINARY_UNSUPPORTED, BINARY_OPT_IN}
            if entry.binary_scan == BINARY_OPT_IN:
                assert entry.hardware_unverified, name


class TestPreferences:
    @staticmethod
    def test_both_switches_default_off():
        defaults = ScanPreferences()
        assert defaults.combined_text is False
        assert defaults.binary_scan is False
        assert defaults.enabled is False
        assert defaults.describe() == "trusted per-channel reads"

    @staticmethod
    def test_an_explicit_switch_is_named():
        assert (
            "combined text scan"
            in ScanPreferences(combined_text=True).describe()
        )
        assert "binary scan" in ScanPreferences(binary_scan=True).describe()

    @staticmethod
    def test_the_trusted_default_reports_each_drivers_own_path():
        for name, entry in DRIVERS.items():
            assert entry.requested(ScanPreferences()) == entry.path, name


class TestRequestedNeverUpgrades:
    @staticmethod
    @pytest.mark.parametrize("name", ["NanoVNA_V2", "LiteVNA64"])
    def test_an_already_combined_driver_is_never_reported_as_opt_in(name):
        entry = DRIVERS[name]
        assert entry.requested(OPT_IN) in {PATH_FIFO, "scan_bin"}
        assert entry.requested(ScanPreferences(combined_text=True)) == PATH_FIFO

    @staticmethod
    def test_a_masked_driver_is_never_reported_as_opt_in():
        masked = ScanCapability(
            driver="masked",
            path=PATH_SCAN_MASK,
            combines_channels=True,
            combined_text_opt_in=True,
        )
        assert masked.requested(OPT_IN) == PATH_SCAN_MASK

    @staticmethod
    def test_a_plain_text_driver_is_offered_the_combined_read():
        entry = DRIVERS["SV4401A"]
        assert entry.requested(ScanPreferences(combined_text=True)) == (
            "combined-text"
        )

    @staticmethod
    def test_an_old_firmware_is_downgraded_to_sweep():
        entry = DRIVERS["NanoVNA"].with_runtime("sweep")
        assert entry.path == PATH_SWEEP
        assert entry.requested(OPT_IN) != "combined-text"
        assert entry.requested(OPT_IN) != "scan_bin"

    @staticmethod
    def test_a_mask_firmware_is_folded_in():
        entry = DRIVERS["NanoVNA"].with_runtime(PATH_SCAN_MASK)
        assert entry.combines_channels
        assert entry.requested(OPT_IN) == PATH_SCAN_MASK


class TestLiveDevices:
    @staticmethod
    def test_the_v2_family_reports_its_own_binary_read():
        for cls in (NanoVNA_V2, LiteVNA64):
            entry = scan_capability_for(object.__new__(cls))
            assert entry.path == PATH_FIFO
            assert entry.binary_transport

    @staticmethod
    def test_a_text_driver_reports_the_path_its_firmware_chose():
        nano = object.__new__(NanoVNA)
        nano.sweep_method = "sweep"
        assert scan_capability_for(nano).path == PATH_SWEEP
        nano.sweep_method = "scan_mask"
        assert scan_capability_for(nano).path == PATH_SCAN_MASK

    @staticmethod
    def test_a_subclass_inherits_its_parents_claim():
        class Custom(NanoVNA):
            pass

        entry = scan_capability_for(object.__new__(Custom))
        assert entry.path == DRIVERS["NanoVNA"].path

    @staticmethod
    def test_the_capability_carries_the_matrix_entry():
        capability = SweepCapability.from_vna(object.__new__(NanoVNA))
        assert capability.scan.path == DRIVERS["NanoVNA"].path
        assert f"reads: {capability.scan.path}" in capability.describe()


class TestUnverifiedList:
    @staticmethod
    def test_every_driver_with_an_optimised_read_is_listed():
        listed = set(hardware_unverified_drivers())
        for name, entry in DRIVERS.items():
            if (
                entry.combined_text_opt_in
                or entry.binary_scan == BINARY_OPT_IN
                or entry.hardware_unverified
            ):
                assert name in listed

    @staticmethod
    def test_the_physical_text_and_binary_drivers_are_all_unverified():
        listed = set(hardware_unverified_drivers())
        for name in (
            "NanoVNA",
            "NanoVNA_H",
            "NanoVNA_H4",
            "NanoVNA_F",
            "NanoVNA_F_V2",
            "NanoVNA_F_V3",
            "JNCRadio_VNA_3G",
            "SV4401A",
            "SV6301A",
            "NanoVNA_V2",
            "LiteVNA64",
        ):
            assert name in listed

    @staticmethod
    def test_drivers_without_an_optimised_path_are_not_blamed():
        listed = set(hardware_unverified_drivers())
        for name in ("AVNA", "TinySA", "TinySA_Ultra", "VNA"):
            assert name not in listed
