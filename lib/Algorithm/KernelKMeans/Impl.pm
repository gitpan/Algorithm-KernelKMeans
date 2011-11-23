package Algorithm::KernelKMeans::Impl;

use namespace::autoclean;

use Carp;
use List::MoreUtils qw/all/;

use Moose::Role;
use Moose::Util::TypeConstraints;
use MooseX::Types::Common::Numeric qw/PositiveOrZeroNum/;
use MooseX::Types::Moose qw/ArrayRef CodeRef HashRef Int/;
use MooseX::Types::Structured qw/Tuple slurpy/;

has 'vectors' => (
  is => 'ro',
  isa => subtype(ArrayRef[ HashRef[PositiveOrZeroNum] ] => where { @$_ > 0 }),
  traits => [qw/Array/],
  handles => +{
    vector => 'get',
    num_vectors => 'count'
  },
  required => 1
);

has 'weights' => (
  is => 'ro',
  isa => ArrayRef[PositiveOrZeroNum],
  lazy => 1,
  traits => [qw/Array/],
  handles => +{
    weight => 'get',
    num_weights => 'count'
  },
  builder => '_build_weights'
);

has 'kernel' => (
  is => 'ro',
  isa => (CodeRef | Tuple[Int, slurpy ArrayRef]),
  predicate => 'has_kernel'
);

# vector index -> vector index -> kernel
# It is assumed that K(x1, x2) = K(x2, x1)
has 'kernel_matrix' => (
  is => 'ro',
  isa => ArrayRef[ ArrayRef[PositiveOrZeroNum] ],
  lazy => 1,
  builder => '_build_kernel_matrix',
  predicate => 'has_kernel_matrix'
);

requires qw/_build_kernel_matrix cluster_indices/;

around '_build_kernel_matrix' => sub {
  my ($orig, $self) = @_;
  my $mat = $self->$orig;
  croak 'Kernel matrix seems too small'
    unless $self->_validate_kernel_matrix($mat);
  return $mat;
};

sub _validate_kernel_matrix {
  my ($self, $mat) = @_;
  my $i = 1;
  @$mat >= $self->num_vectors and all { @$_ >= $i++ } @$mat;
}

sub _build_weights { [ (1.0) x shift->num_vectors ] }

sub BUILD {
  my $self = shift;

  if ($self->num_vectors != $self->num_weights) {
    croak 'Array "vectors" and "weights" must be same size';
  }

  unless ($self->has_kernel or $self->has_kernel_matrix) {
    croak 'Either of "kernel" or "kernel_matrix" is required';
  }

  if ($self->has_kernel and $self->has_kernel_matrix) {
    croak 'Argument "kernel" and "kernel_matrix" are exclusive';
  }

  if ($self->has_kernel_matrix
        and not $self->_validate_kernel_matrix($self->kernel_matrix)) {
    croak 'Kernel matrix seems too small';
  }
};

sub vectors_of {
  my ($self, $indices) = @_;
  [ map { $self->vector($_) } @$indices ];
}

sub weights_of {
  my ($self, $indices) = @_;
  [ map { $self->weight($_) } @$indices ];
}

sub run {
  my ($self, %opts) = @_;
  my $clusters = $self->cluster_indices(%opts);
  [ map { $self->vectors_of($_) } @$clusters ];
}

1;

__END__;

=head1 NAME

Algoirthm::KernelKmeans::Impl

=head1 SYNOPSIS

  package Algorithm::KernelKmeans::YetAnotherImplClass
  
  use Moose;
  with qw/Algorithm::KernelKMeans::Impl/;

=head1 REQUIRES

=head2 _build_kernel_matrix

=head2 cluster_indices

=head1 METHODS

=head2 run

=head2 vectors_of

=head2 weights_of

=head1 AUTHOR

Koichi SATOH E<lt>sekia@cpan.orgE<gt>

=head1 SEE ALSO

L<Algorithm::KernelKMeans::PP>

=cut
