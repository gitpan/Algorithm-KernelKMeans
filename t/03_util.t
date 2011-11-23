use strict;
use warnings;

use ExtUtils::testlib;
use FindBin;
use List::MoreUtils qw/zip/;
use Test::More;

use Algorithm::KernelKMeans::Util qw/centroid
                                     inner_product
                                     diff_vector
                                     euclidean_distance/;

open my $vectors, '<', "$FindBin::Bin/vectors.txt" or die $!;
my @vectors = map {
  my @vals = split /\s+/;
  my @keys = 0 .. $#vals;
  +{ zip @keys, @vals };
} <$vectors>;

open my $kmat, '<', "$FindBin::Bin/kernels.txt" or die $!;
my @kernel_matrix = map { [ split /\s+/ ] } <$kmat>;

{
  my $centroid = centroid([
    +{ foo => 1, bar => 3, baz => 5 },
    +{ foo => 2, bar => 4, baz => 6 },
    +{ foo => 1, bar => 2, baz => 4 }
  ]);
  is_deeply $centroid, +{ foo => 4/3, bar => 9/3, baz => 15/3 };
}

{
  my $centroid = centroid([
    +{ foo => 1, bar => -3, baz => 5 },
    +{ hoge => 2, fuga => 4, piyo => 6 },
    +{ foo => 1, bar => -2, quux => 4 }
  ]);
  is_deeply $centroid, +{ foo => 2/3, bar => -5/3, baz => 5/3,
                          hoge => 2/3, fuga => 4/3, piyo => 6/3, quux => 4/3 };
}

{
  my $vec1 = +{ x => 3, y => 1 };
  my $vec2 = +{ x => 2, z => 1 };
  my $diff = diff_vector $vec1, $vec2;
  is_deeply $diff, +{ x => 1, y => 1, z => -1 };
}

{
  my $vec1 = +{ x => 2, y => 3 };
  my $vec2 = +{ x => -2 };
  my $dist = euclidean_distance $vec1, $vec2;
  is $dist, 5;
}

done_testing;
