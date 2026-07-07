#!/usr/bin/perl
use strict;
use warnings;

my $root = "$ENV{HOME}/openfoam/RGP-13/of-core-mods/fvMeshTopoChangers";
my $cpath = "$root/refiner/refiner_fvMeshTopoChanger.C";
my $mpath = "$root/Make/files";

# ---- 1) Retarget Make/files LIB to FOAM_USER_LIBBIN ----
{
    local $/; open my $fh, '<', $mpath or die "open $mpath: $!";
    my $m = <$fh>; close $fh;
    my $old = 'LIB = $(FOAM_LIBBIN)/libfvMeshTopoChangers';
    my $new = 'LIB = $(FOAM_USER_LIBBIN)/libfvMeshTopoChangers';
    die "Make/files LIB line not found\n" unless index($m, $old) >= 0;
    $m =~ s/\Q$old\E/$new/;
    open my $out, '>', $mpath or die; print $out $m; close $out;
    print "Make/files patched -> FOAM_USER_LIBBIN\n";
}

# ---- 2) Patch the .C ----
my $c;
{ local $/; open my $fh, '<', $cpath or die "open $cpath: $!"; $c = <$fh>; close $fh; }

# 2a) add include
my $inc_anchor = '#include "addToRunTimeSelectionTable.H"';
die "include anchor not found\n" unless index($c, $inc_anchor) >= 0;
die "include already present\n" if index($c, '#include "polyDistributionMap.H"') >= 0;
$c =~ s/\Q$inc_anchor\E/$inc_anchor\n#include "polyDistributionMap.H"/;

# 2b) replace distribute body
my $old_block = <<'END_OLD';
void Foam::fvMeshTopoChangers::refiner::distribute
(
    const polyDistributionMap& map
)
{
    // Redistribute the mesh cutting engine
    meshCutter_.distribute(map);
}
END_OLD

my $new_block = <<'END_NEW';
void Foam::fvMeshTopoChangers::refiner::distribute
(
    const polyDistributionMap& map
)
{
    // Redistribute the mesh cutting engine (cellLevel/pointLevel/history)
    meshCutter_.distribute(map);

    // Redistribute the protected-cell marking to the new decomposition.
    // Upstream refiner::distribute only handles meshCutter_; protectedCells_
    // is sized to the OLD local cell count and would be stale (wrong size /
    // mis-indexed) after redistribution, corrupting the next (un)refinement.
    if (protectedCells_.size())
    {
        labelList protectedCell(protectedCells_.size());
        forAll(protectedCell, celli)
        {
            protectedCell[celli] = protectedCells_.get(celli);
        }

        map.distributeCellData(protectedCell);

        PackedBoolList newProtectedCell(protectedCell.size());
        forAll(protectedCell, celli)
        {
            newProtectedCell.set(celli, protectedCell[celli]);
        }
        protectedCells_.transfer(newProtectedCell);
    }
}
END_NEW

die "distribute() block not matched exactly\n" unless index($c, $old_block) >= 0;
my $cnt = ($c =~ s/\Q$old_block\E/$new_block/);
die "expected 1 replacement, got $cnt\n" unless $cnt == 1;

open my $out, '>', $cpath or die; print $out $c; close $out;
print "refiner_fvMeshTopoChanger.C patched: include + protectedCells_ redistribution\n";
