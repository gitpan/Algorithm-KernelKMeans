use strict;
use warnings;

use ExtUtils::testlib;
use FindBin;
use Test::More;

use lib $FindBin::Bin; # for Algorithm::NaiveKmeans

use Algorithm::KernelKMeans;
use Algorithm::KernelKMeans::Util qw/generate_polynominal_kernel/;
use Algorithm::NaiveKMeans;

diag 'This test may take some minutes';
diag "Using $Algorithm::KernelKMeans::IMPLEMENTATION implementation";

open my $vectors , '<', "$FindBin::Bin/vectors.txt" or die $!;
my @vertices = map { [ split /\s+/ ] } (<$vectors>)[0 .. 255];

sub sort_cluster {
  [ sort {
    $a->[0] <=> $b->[0] or $a->[1] <=> $b->[1] or $a->[2] <=> $b->[2]
  } @{ +shift } ]
}

my $kkm = Algorithm::KernelKMeans->new(
  vertices => \@vertices,
  kernel => generate_polynominal_kernel(0, 1) # just inner product
);
my $kkm_clusters = $kkm->run(k => 6, shuffle => 0);
my @kkm_clusters = map { sort_cluster $_ } @$kkm_clusters;

my $nkm = Algorithm::NaiveKMeans->new(vertices => \@vertices);
my @nkm_clusters = map { sort_cluster $_ } @{ $nkm->run(k => 6, shuffle => 0) };

is_deeply \@kkm_clusters, \@nkm_clusters,
  'WKKM with uniform weights and identity kernel is equivalant to naive KM';

done_testing;
