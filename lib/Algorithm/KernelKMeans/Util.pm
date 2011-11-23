package Algorithm::KernelKMeans::Util;

use 5.010;
use strict;
use warnings;

use Attribute::Constant;
use Exporter::Lite;
use List::Util qw/sum/;

our @EXPORT_OK = qw/centroid
                    diff_vector
                    inner_product
                    euclidean_distance
                    $KERNEL_POLYNOMINAL $KERNEL_GAUSSIAN $KERNEL_SIGMOID
                    $INITIALIZE_SIMPLE $INITIALIZE_SHUFFLE $INITIALIZE_KKZ/;

our $KERNEL_POLYNOMINAL : Constant(0);
our $KERNEL_GAUSSIAN    : Constant(1);
our $KERNEL_SIGMOID     : Constant(2);

our $INITIALIZE_SIMPLE  : Constant(0);
our $INITIALIZE_SHUFFLE : Constant(1);
our $INITIALIZE_KKZ     : Constant(2);

sub centroid {
  my $cluster = shift;
  my %centroid;
  for my $vector (@$cluster) {
    while (my ($key, $val) = each %$vector) {
      $centroid{$key} //= 0;
      $centroid{$key} += $val;
    }
  }
  for my $key (keys %centroid) { $centroid{$key} /= @$cluster }
  return \%centroid;
}

sub inner_product {
  my ($x1, $x2) = @_;
  my @common_keys = grep { exists $x2->{$_} } keys %$x1;
  return 0 if @common_keys == 0;
  sum map { $x1->{$_} * $x2->{$_} } @common_keys;
}

sub diff_vector {
  my ($v, $u) = @_;
  my %tmp; @tmp{keys %$v, keys %$u} = ();
  my %diff = map {
    my ($e1, $e2) = (($v->{$_} // 0), ($u->{$_} // 0));
    ($_ => $e1 - $e2);
  } keys %tmp;
  \%diff;
}

sub euclidean_distance {
  my ($v, $u) = @_;
  my $sub = diff_vector($v, $u);
  sqrt inner_product($sub, $sub);
}

1;

__END__

=head1 NAME

Algorithm::KernelKMeans::Util

=head1 DESCRIPTION

This module provides some constants and functions suitable to use with C<Algorithm::KernelKMeans>.

=head1 CONSTANTS

Constants listed below represent kernels/initializers, which some methods require.
It's recommended to use these constants instead of code reference, because clusterer implementations might have its own kernel/initializer code.
So code references should be used iff there's no equivalent constants.

All constants are C<import()>able.

=head2 $KERNEL_POLYNOMINAL

Polynominal kernel. Takes 2 parameters ($l, $p) then will be formed like C<K(x1, x2) = ($l + x1 . x2)^$p, where "x1 . x2" represents inner product>.

=head2 $KERNEL_GAUSSIAN

Gaussian kernel. Takes 1 parameter ($sigma).

C<K(x1, x2) = exp(-||x1 - x2||^2 / (2 * $sigma)^2)>

=head2 $KERNEL_SIGMOID

Sigmoid kernel. Takes 2 parameters ($s, $theta).

C<K(x1, x2) = tanh($s * (x1 . x2) + $theta)>

=head2 $INITIALIZE_SIMPLE

=head2 $INITIALIZE_SHUFFLE

=head2 $INITIALIZE_KKZ

=head1 FUNCTIONS

This module exports nothing by default. You can C<import> functions below:

=head2 centroid($cluster)

Takes array ref of vectors and returns centroid vector of the cluster.

=head2 inner_product($v, $u)

Calculates inner product of C<$v> and C<$u>.

=head2 euclidean_distance($v, $u)

Computes euclidean distance between C<$v> and C<$u>.

=head1 AUTHOR

Koichi SATOH E<lt>r.sekia@gmail.comE<gt>

=cut
