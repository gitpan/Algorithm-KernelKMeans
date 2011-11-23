use strict;
use warnings;

use Data::Dumper;
use ExtUtils::testlib;
use FindBin;
use List::MoreUtils qw/all zip/;
use Test::More;
use Test::Exception;

use lib $FindBin::Bin; # for Algorithm::NaiveKmeans

use Algorithm::KernelKMeans::PP;
use Algorithm::KernelKMeans::Util qw/$KERNEL_POLYNOMINAL
                                     $KERNEL_GAUSSIAN
                                     $KERNEL_SIGMOID
                                     $INITIALIZE_SIMPLE
                                     inner_product/;
use Algorithm::NaiveKMeans;

diag 'This test may take some minutes';

# 3D vectors
open my $vectors , '<', "$FindBin::Bin/vectors.txt" or die $!;
my @vectors = map {
  my @vals = split /\s+/;
  my @keys = 0 .. $#vals;
  +{ zip @keys, @vals };
} <$vectors>;

my @weights = (1.0) x @vectors;

# Precomputed kernel matrix for @vectors
open my $kmat, '<', "$FindBin::Bin/kernels.txt" or die $!;
my @kernel_matrix = map { [ split /\s+/ ] } <$kmat>;

subtest '"vectors" and "kernel" (or "kernel_matrix") is required' => sub {
  dies_ok { Algorithm::KernelKMeans::PP->new };
  dies_ok { Algorithm::KernelKMeans::PP->new(vectors => []) };
  dies_ok { Algorithm::KernelKMeans::PP->new(vectors => \@vectors) };
  lives_ok {
    Algorithm::KernelKMeans::PP->new(
      vectors => \@vectors,
      kernel => [$KERNEL_POLYNOMINAL => (1, 2)]
    );
  };
  lives_ok {
    Algorithm::KernelKMeans::PP->new(
      vectors => \@vectors,
      kernel_matrix => \@kernel_matrix
    );
  };
  dies_ok {
    Algorithm::KernelKMeans::PP->new(
      vectors => \@vectors,
      kernel => [$KERNEL_POLYNOMINAL => (1, 2)],
      kernel_matrix => \@kernel_matrix
    );
  } '"kernel" and "kernel_matrix" are exclusive';
  done_testing;
};

subtest '"vectors" and "weights" must be same size' => sub {
  dies_ok {
    Algorithm::KernelKMeans::PP->new(
      vectors => \@vectors,
      weights => [1, 1, 1],
      kernel => [$KERNEL_POLYNOMINAL => (1, 2)]
    );
  };
  lives_ok {
    Algorithm::KernelKMeans::PP->new(
      vectors => \@vectors,
      weights => [0 .. $#vectors],
      kernel => [$KERNEL_POLYNOMINAL => (1, 2)]
    );
  };

  done_testing;
};

{
  my $km1 = Algorithm::KernelKMeans::PP->new(
    vectors => \@vectors,
    kernel => [$KERNEL_POLYNOMINAL => (1, 2)] # recommended style
  )->kernel_matrix;
  my $km2 = Algorithm::KernelKMeans::PP->new(
    vectors => \@vectors,
    kernel => Algorithm::KernelKMeans::PP::generate_polynominal_kernel(1, 2)
  )->kernel_matrix;
  is_deeply $km1, $km2, 'Kernel function can be specified as coderef or tuple';
}

subtest 'Polynominal kernel needs 2 parameters' => sub {
  lives_ok {
    Algorithm::KernelKMeans::PP->new(
      vectors => \@vectors,
      kernel => [$KERNEL_POLYNOMINAL => (1, 2)]
    )->kernel_matrix;
  };
  dies_ok {
    Algorithm::KernelKMeans::PP->new(
      vectors => \@vectors,
      kernel => [$KERNEL_POLYNOMINAL => (1)]
    )->kernel_matrix;
  };
  done_testing;
};

subtest 'Gaussian kernel needs 1 parameter' => sub {
  lives_ok {
    Algorithm::KernelKMeans::PP->new(
      vectors => \@vectors,
      kernel => [$KERNEL_GAUSSIAN => (1)]
    )->kernel_matrix;
  };
  dies_ok {
    Algorithm::KernelKMeans::PP->new(
      vectors => \@vectors,
      kernel => [$KERNEL_GAUSSIAN => ()]
    )->kernel_matrix;
  };
  done_testing;
};

subtest 'Sigmoid kernel needs 2 parameters' => sub {
  lives_ok {
    Algorithm::KernelKMeans::PP->new(
      vectors => \@vectors,
      kernel => [$KERNEL_SIGMOID => (1, 2)]
    )->kernel_matrix;
  };
  dies_ok {
    Algorithm::KernelKMeans::PP->new(
      vectors => \@vectors,
      kernel => [$KERNEL_SIGMOID => (1)]
    )->kernel_matrix;
  };
  done_testing;
};

subtest 'Checks kernel matrix size' => sub {
  dies_ok {
    Algorithm::KernelKMeans::PP->new(
      vectors => \@vectors,
      kernel_matrix => [ @kernel_matrix[0 .. 31] ]
    );
  };
  dies_ok {
    Algorithm::KernelKMeans::PP->new(
      vectors => \@vectors,
      kernel_matrix => [ ([1, 3, 5]) x @vectors ]
    );
  };
  done_testing;
};

sub kernels {
  my ($kernel, $vectors) = @_;
  [ map {
    my $i = $_;
    map {
      my ($v, $u) = ($vectors->[$i], $vectors->[$_]);
      $kernel->($v, $u);
    } 0 .. $i;
  } 0 .. $#$vectors ];
}

{
  no strict qw/refs/;
  for my $sym (qw/generate_polynominal_kernel
                  generate_sigmoid_kernel
                  generate_gaussian_kernel
                  init_clusters_simple
                  init_clusters_shuffle
                  init_clusters_kkz/) {
    *$sym = \&{ "Algorithm::KernelKMeans::PP::$sym" };
  }
}

subtest 'Kernels' => sub {
  my $inner_products = kernels(\&inner_product, \@vectors);
  my $simple_poly_kernel = generate_polynominal_kernel(0, 1);
  my $simple_poly_kernels = kernels($simple_poly_kernel, \@vectors);
  is_deeply $simple_poly_kernels, $inner_products;

  my $poly_kernel = generate_polynominal_kernel(1, 2);
  my $poly_kernels = kernels($poly_kernel, \@vectors);
  ok +(all { $_ > 0 } @$poly_kernels), 'Polynominal kernel is positive definite';

  my $gaus_kernel = generate_gaussian_kernel(3);
  my $gaus_kernels = kernels($gaus_kernel, \@vectors);
  ok +(all { $_ > 0 } @$gaus_kernels), 'Gaussian kernel is positive definite';

  my $sigm_kernel = generate_sigmoid_kernel(1, 0);
  my $sigm_kernels = kernels($sigm_kernel, \@vectors);
  ok +(all { $_ >= 0 } @$sigm_kernels), 'Sigmoid kernel is (almost) semi-positive definite';

  done_testing;
};

subtest 'Cluster initializers' => sub {
  my $clus1 = init_clusters_simple(6, \@vectors, \@weights, \@kernel_matrix);
  my $clus2 = init_clusters_simple(6, \@vectors, \@weights, \@kernel_matrix);
  ok +(all { @$_ > 0 } @$clus1);
  is_deeply $clus1, $clus2;

  my $clus3 = init_clusters_shuffle(6, \@vectors, \@weights, \@kernel_matrix);
  my $clus4 = init_clusters_shuffle(6, \@vectors, \@weights, \@kernel_matrix);
  ok +(all { @$_ > 0 } @$clus3);
  isnt Dumper($clus3), Dumper($clus4);

  my $clus5 = init_clusters_kkz(6, \@vectors, \@weights, \@kernel_matrix);
  ok +(all { @$_ > 0 } @$clus5);

  done_testing;
};

sub sort_cluster {
  [ sort {
    $a->{0} <=> $b->{0} or $a->{1} <=> $b->{1} or $a->{2} <=> $b->{2}
  } @{ +shift } ]
}

{
  my $kkm = Algorithm::KernelKMeans::PP->new(
    vectors => \@vectors,
    kernel => [$KERNEL_POLYNOMINAL => (0, 1)] # just inner product
  );
  my $kkm_clusters = $kkm->run(k => 6, initializer => $INITIALIZE_SIMPLE);
  my @kkm_clusters = map { sort_cluster $_ } @$kkm_clusters;

  my $nkm = Algorithm::NaiveKMeans->new(vectors => \@vectors);
  my $nkm_clusters = $nkm->run(k => 6, shuffle => 0);
  my @nkm_clusters = map { sort_cluster $_ } @$nkm_clusters;

  is_deeply \@kkm_clusters, \@nkm_clusters,
    'WKKM with uniform weights and identity kernel is equivalant to naive KM';
}

{
  my $kkm = Algorithm::KernelKMeans::PP->new(
    vectors => \@vectors,
    kernel => [$KERNEL_POLYNOMINAL => (1, 2)]
  );

  dies_ok {
    $kkm->run;
  } '"k" is required';

  dies_ok {
    $kkm->run(k =>  6, k_min => 10);
  } '"k_min" must be less than or equal to "k"';

  dies_ok {
    $kkm->run(k => 6, foo => 'bar');
  } 'Unkown parameter should be error';

  my @clusters1 = map { sort_cluster $_ } @{ $kkm->run(k => 6) };
  my @clusters2 = map { sort_cluster $_ } @{ $kkm->run(k => 6) };
  is_deeply \@clusters1, \@clusters2,
    'WKKM with same initial cluster is deterministic';
}

done_testing;
