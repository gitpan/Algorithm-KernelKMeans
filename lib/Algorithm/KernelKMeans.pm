package Algorithm::KernelKMeans;

use 5.010;
use namespace::autoclean;
use version;
use Moose;
use UNIVERSAL::require;

our $VERSION = '0.03_02';

our $IMPLEMENTATION;
for my $impl (qw/XS PP/) {
  my $impl_class = __PACKAGE__ . '::' . $impl;
  if ($impl_class->require) {
    next if $impl eq 'XS'
      and version->parse($impl_class->VERSION) < version->parse('0.02_02');
    $IMPLEMENTATION = $impl_class;
    extends $IMPLEMENTATION;
    last;
  }
}

__PACKAGE__->meta->make_immutable;

__END__

=head1 NAME

Algorithm::KernelKMeans - Weighted kernel k-means clusterer

=head1 SYNOPSIS

  use Algorithm::KernelKMeans;
  use Algorithm::KernelKMeans::Util qw/$KERNEL_POLYNOMINAL/;
  use List::MoreUtils qw/zip/;
  use Try::Tiny;
  
  my @vectors = map {
    my @values = split /\s/;
    my @keys = 0 .. $#values;
    +{ zip @keys, @values };
  } (<>);
  my $wkkm = Algorithm::KernelKMeans->new( # default weights are 1
    vectors => \@vectors,
    kernel => [$KERNEL_POLYNOMINAL => (1, 2)] # K(x1, x2) = (1 + x1x2)^2
  );
  
  try {
    my $clusters = $wkkm->run(k => 6);
    for my $cluster (@$clusters) {
      ...
    }
  } catch {
    # during iteration, number of clusters became less than k_min
    if (/number of clusters/i) { ... }
  }

=head1 DESCRIPTION

C<Algorithm::KernelKMeans> provides weighted kernel k-means vector clusterer.

Note that this is a very early release. All APIs may be changed incompatibly.

=head2 IMPLEMENTATION

This class is just a placeholder. Implementation code is in other class and this class just inherits it.

Currently there are 2 implementations: L<Algorithm::KernelKMeans::PP> and L<Algorithm::KernelKMeans::XS>.

C<$Algorithm::KernelKMeans::IMPLEMENTATION> indicates which implementation is loaded.

Both of these implements same interface (documented below) and C<Algorithm::KernelKMeans> uses faster (XS) implementation if it's available.
So it's not necessary usually to use the classes directly tough, you can do it if you want.

=head1 METHODS

=head2 new(%opts)

Constructor. you can specify options below:

=head3 vectors

Required. Array of vectors.
Each vector is represented as an hash of positive real numbers.

e.g.:

 my $wkkm = Algorithm::KernelKMeans->new(
   vectors => [ +{ prop1 => 229, prop2 => 151, prop3 =>  42 },
                 +{ prop1 =>  61, prop2 => 151, prop4 => 251 },
                 +{ prop2 =>  11, prop3 => 120, prop4 =>  55 } ],
   kernel => [$KERNEL_POLYNOMINAL => (1, 2)]
 );

=head3 weights

Array of positive real numbers. Defaults to list of 1s.

=head3 kernel

Function projects 2 vectors into higher dimentional space and computes inner product.

Kernel function can be specified as a tuple or a code reference.

Tuple is formed with descriptor and parameter(s). For example:

  [$KERNEL_POLYNOMINAL => (1, 2)]

C<$KERNEL_POLYNOMINAL> is a descriptor. And rest of the elements are parameters.

L<Algorithm::KernelKMeans::Util> has some descriptors for some popular kernel functions.

=head3 kernel_matrix

2D array of kernel values.

A matrix whose element at (i, j) is K(xi, xj) where i >= j.
This is derived automatically from C<kernel> by default, however you can specify it manually if you already have it.

Note that the clusterer only uses lower triangle part of the matrix.
So it is not necessary for the matrix to have element at (i, j) where i < j.

Note that C<kernel> and C<kernel_matrix> are exclusive and either of these is required.

=head2 run(%opts)

Executes clustering. Return value is an array ref of clusters.

=head3 k

Required. (maximum) number of clusters.

=head3 k_min

Some clusters may be empty during clustering.
In the case, the clusterer just removes the empty clusters and checks number of rest clusters. If it is less than C<k_min>, the clusterer throws an error.

Default is same as C<k>.

=head3 initializer

Specifies cluster initializing method.
By default, the clusterer initializes clusters using KKZ, which is known as a good initializing procedure.

You can C<import> some initializer descriptors from C<Algorithm::KernelKMeans::Util>.

=head3 converged

Function predicates that clustering is converged.
Iteration is broken off and returns result when the predicate returns true.

For each iteration, 2 values will be specified:
objective function value of current clusters and new clusters' one.
As clusters converges, the value decreases.

Default predicate just checks if these 2 values are equal.

=head2 cluster_indices(%opts)

This method is similar to C<run>, but returns clusters contain indices instead of vectors.

=head1 AUTHOR

Koichi SATOH E<lt>sekia@cpan.orgE<gt>

=head1 SEE ALSO

L<Algorithm::KernelKMeans::PP> - Default implementation

L<Algorithm::KernelKMeans::XS> - Yet another implementation. Fast!

=head1 LICENSE

The MIT License

Copyright (C) 2010 by Koichi SATOH

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

=cut
