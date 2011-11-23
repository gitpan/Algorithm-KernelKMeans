package Algorithm::KernelKMeans::PP;

use 5.010;
use namespace::autoclean;

use Carp;
use List::Util qw/min reduce sum shuffle/;
use List::MoreUtils qw/natatime pairwise/;
use Moose;
use POSIX qw/floor tanh/;

use Algorithm::KernelKMeans::Util qw/$KERNEL_POLYNOMINAL
                                     $KERNEL_GAUSSIAN
                                     $KERNEL_SIGMOID
                                     $INITIALIZE_SIMPLE
                                     $INITIALIZE_SHUFFLE
                                     $INITIALIZE_KKZ
                                     inner_product
                                     euclidean_distance/;

with qw/Algorithm::KernelKMeans::Impl/;

sub init_clusters_simple {
  my ($k, $vectors) = @_;
  my $nvectors = @$vectors;
  _init_clusters($k, [ 0 .. $nvectors - 1 ]);
}

sub init_clusters_shuffle {
  my ($k, $vectors) = @_;
  my $nvectors = @$vectors;
  _init_clusters($k, [ shuffle 0 .. $nvectors - 1 ]);
}

sub _init_clusters {
  my ($k, $indices) = @_;
  my $cluster_size = floor($#$indices / $k);
  my $iter = natatime $cluster_size, @$indices;
  my @clusters;
  while (my @cluster = $iter->()) { push @clusters, \@cluster }
  if (@{ $clusters[-1] } < $cluster_size) {
    my $last_cluster = pop @clusters;
    push @{ $clusters[-1] }, @$last_cluster;
  }
  return \@clusters;
}

sub init_clusters_kkz {
  my ($k, $vectors) = @_;
  my $nvectors = @$vectors;
  my @rep_vectors; # cluster representation vectors

  my $first_vector = reduce {
    $a->[1] > $b->[1] ? $a : $b
  } map {
    my $vector = $vectors->[$_];
    [ $_ => inner_product($vector, $vector) ];
  } 0 .. $nvectors - 1;
  push @rep_vectors, $first_vector->[0];

  until (@rep_vectors == $k) {
    my @ds = map {
      my $vector = $vectors->[$_];
      my $min = min map {
        euclidean_distance($vector, $vectors->[$_])
      } @rep_vectors;
      [ $_ => $min ];
    } 0 .. $nvectors - 1;
    my $rep_vector = reduce { $a->[1] > $b->[1] ? $a : $b } @ds;
    push @rep_vectors, $rep_vector->[0];
  }

  my @clusters;
  for my $i (0 .. $nvectors - 1) {
    my $vector = $vectors->[$i];
    my @ds = map {
      my $rep_vector_idx = $rep_vectors[$_];
      [ $_ => euclidean_distance($vector, $vectors->[$rep_vector_idx]) ];
    } 0 .. $#rep_vectors;
    my $nearest = reduce { $a->[1] <= $b->[1] ? $a : $b } @ds;
    push @{ $clusters[$nearest->[0]] }, $i;
  }
  return \@clusters;
}

sub generate_polynominal_kernel {
  my ($l, $p) = @_;
  sub {
    my ($x1, $x2) = @_;
    ($l + inner_product($x1, $x2)) ** $p
  }
}

sub generate_gaussian_kernel {
  my $sigma = shift;
  my $numer = 2 * ($sigma ** 2);
  sub {
    my ($x1, $x2) = @_;
    my %tmp; @tmp{keys %$x1, keys %$x2} = ();
    my $norm = sqrt sum map {
      my ($e1, $e2) = (($x1->{$_} // 0), ($x2->{$_} // 0));
      ($e1 - $e2) ** 2;
    } keys %tmp;
    exp(-$norm / $numer);
  }
}

sub generate_sigmoid_kernel {
  my ($s, $theta) = @_;
  sub {
    my ($x1, $x2) = @_;
    tanh($s * inner_product($x1, $x2) + $theta);
  }
}

sub _build_kernel_matrix {
  my $self = shift;

  my $kernel;
  if (ref $self->kernel eq 'CODE') {
    $kernel = $self->kernel;
  } else {
    my ($kernel_desc, @params) = @{ $self->kernel };
    given ($kernel_desc) {
      when ($KERNEL_POLYNOMINAL) {
        croak 'Too few parameters' if @params < 2;
        $kernel = generate_polynominal_kernel(@params);
      }
      when ($KERNEL_GAUSSIAN) {
        croak 'Too few parameters' if @params < 1;
        $kernel = generate_gaussian_kernel(@params);
      }
      when ($KERNEL_SIGMOID) {
        croak 'Too few parameters' if @params < 2;
        $kernel = generate_sigmoid_kernel(@params);
      }
      default { croak 'Unknown kernel function' }
    }
  }

  my @matrix = map {
    my $i = $_;
    [ map {
      my $j = $_;
      $kernel->($self->vector($i), $self->vector($j));
    } 0 .. $i ];
  } 0 .. $self->num_vectors - 1;
  return \@matrix;
}

sub init_clusters {
  my ($self, $init, $k) = @_;
  $init->($k, $self->vectors, $self->weights, $self->kernel_matrix);
}

sub total_weight_of {
  my ($self, $cluster) = @_;
  sum @{ $self->weights_of($cluster) };
}

sub step {
  my ($self, $clusters, $norms) = @_;
  my @new_clusters = map { [] } 0 .. $#$clusters;
  for my $i (0 .. $self->num_vectors - 1) {
    my ($nearest) = sort { $a->[1] <=> $b->[1] } map {
      [ $_ => $norms->[$i][$_] ]
    } 0 .. $#$clusters;
    push @{ $new_clusters[$nearest->[0]] }, $i;
  }
  return [ grep { @$_ != 0 } @new_clusters ];
}

sub compute_score {
  my ($self, $clusters, $norms) = @_;
  my $score = 0;
  for my $cluster_idx (0 .. $#$clusters) {
    my $cluster = $clusters->[$cluster_idx];
    $score += sum map {
      $self->weight($_) * $norms->[$_][$cluster_idx]
    } @$cluster;
  }
  return $score;
}

sub compute_norms {
  my ($self, $clusters) = @_;
  my @total_weights = map { $self->total_weight_of($_) } @$clusters;

  my @term3_denoms = map {
    $self->_norm_term3_denom_of($_)
  } @$clusters;
  my @term3s = pairwise { $a / ($b ** 2) } @term3_denoms, @total_weights;

  my @norms = map {
    my $i = $_;
    my $term1 = $self->kernel_matrix->[$i][$i];
    [ map {
      my $cluster_idx = $_;
      my $cluster = $clusters->[$cluster_idx];
      my $total_weight = $total_weights[$cluster_idx];

      my $weights = $self->weights_of($cluster);
      my $term2 = -2 * sum(pairwise {
        my ($s, $t) = $i >= $a ? ($i, $a) : ($a, $i);
        $self->kernel_matrix->[$s][$t] * $b
      } @$cluster, @$weights) / $total_weight;
      my $term3 = $term3s[$cluster_idx];

      $term1 + $term2 + $term3;
    } 0 .. $#$clusters ]
  } 0 .. $self->num_vectors - 1;
  return \@norms;
}

sub _norm_term3_denom_of {
  my ($self, $cluster) = @_;
  sum map {
    my $i = $_;
    map {
      my ($s, $t) = $i >= $_ ? ($i, $_) : ($_, $i);
      $self->weight($s) * $self->weight($t) * $self->kernel_matrix->[$s][$t];
    } @$cluster;
  } @$cluster;
}

sub cluster_indices {
  my ($self, %opts) = @_;
  my $k = delete $opts{k} // croak 'Missing required parameter "k"';
  my $k_min = delete $opts{k_min} // $k;
  croak '"k_min" must be less than or equal to "k"' if $k_min > $k;
  my $converged = delete $opts{converged} // sub {
    my ($score, $new_score) = @_;
    $score == $new_score;
  };

  my $init = delete $opts{initializer} // $INITIALIZE_KKZ;
  unless (ref $init) {
    given ($init) {
      when ($INITIALIZE_SIMPLE) { $init = \&init_clusters_simple }
      when ($INITIALIZE_SHUFFLE) { $init = \&init_clusters_shuffle }
      when ($INITIALIZE_KKZ) { $init = \&init_clusters_kkz }
      default { croak 'Unknown initializer' }
    }
  }

  if (keys %opts) {
    my $missings = join ', ', map { qq/"$_"/ } sort keys %opts;
    croak "Unknown argument(s): $missings";
  }

  # cluster index -> [vector index]
  my $clusters = $self->init_clusters($init, $k);
  # vector index -> cluster index -> norm
  my $norms = $self->compute_norms($clusters);
  my $score;
  my $new_score = $self->compute_score($clusters, $norms);
  do {
    $clusters = $self->step($clusters, $norms);
    croak "Number of clusters became less than k_min=$k_min"
      if @$clusters < $k_min;
    $norms = $self->compute_norms($clusters);
    $score = $new_score;
    $new_score = $self->compute_score($clusters, $norms);
  } until $converged->($score, $new_score);

  return $clusters;
}

__PACKAGE__->meta->make_immutable;

__END__

=head1 NAME

Algorithm::KernelKMeans::PP

=head1 SYNOPSIS

  use Algorithm::KernelKMeans::PP;

=head1 DESCRIPTION

This class is a pure-Perl implementation of weighted kernel k-means algorithm.

L<Algorithm::KernelKMeans> inherits this class by default.

=head1 AUTHOR

Koichi SATOH E<lt>sekia@cpan.orgE<gt>

=head1 SEE ALSO

L<Algorithm::KernelKMeans>

L<Algorithm::KernelKMeans::XS> - Yet another implementation. Fast!

=head1 LICENSE

The MIT License

Copyright (C) 2010 by Koichi SATOH

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

=cut
