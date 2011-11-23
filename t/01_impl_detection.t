use strict;
use warnings;

use ExtUtils::testlib;
use Test::More;

use Algorithm::KernelKMeans;

my $impl = $Algorithm::KernelKMeans::IMPLEMENTATION;
use_ok $impl;
isa_ok 'Algorithm::KernelKMeans' => $impl;

done_testing;
