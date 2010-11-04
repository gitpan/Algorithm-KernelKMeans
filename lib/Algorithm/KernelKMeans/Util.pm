package Algorithm::KernelKMeans::Util;

use strict;
use warnings;

use Exporter::Lite;
use List::Util qw/sum/;
use List::MoreUtils qw/pairwise/;

our @EXPORT_OK = qw/centroid generate_polynominal_kernel/;

sub centroid {
  my $cluster = shift;
  my @sum = (0) x @{ $cluster->[0] };
  for my $vertex (@$cluster) { @sum = pairwise { $a + $b } @sum, @$vertex }
  [ map { $_ / @$cluster } @sum ];
}

sub generate_polynominal_kernel {
  my ($l, $p) = @_;
  $l //= 1;
  $p //= 2;
  sub {
    my ($x1, $x2) = @_;
    my $inner_product = sum pairwise { $a * $b } @$x1, @$x2;
    ($l + $inner_product) ** $p
  }
}

1;

__END__

=head1 NAME

Algorithm::KernelKMeans::Util

=head1 DESCRIPTION

This module provides some utility functions suitable to use with C<Algorithm::KernelKMeans>.

=head1 FUNCTIONS

This module exports nothing by default. You can C<import> functions below:

=head2 centroid($cluster)

Takes array ref of vertices and returns centroid vector of the cluster.

=head2 generate_polynominal_kernel([$l = 1], [$p = 2])

Generates a polynominal kernel function and returns it.

The generated kernel function will be formed K(x1, x2) = ($l + x1 . x2)^$p ("x1 . x2" means inner product).

=head1 AUTHOR

Koichi SATOH E<lt>r.sekia@gmail.comE<gt>

=cut
