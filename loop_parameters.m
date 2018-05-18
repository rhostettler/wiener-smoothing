% Housekeeping
clear variables;
addpath ~/Projects/smc-mcmc/src/

%%
Mfs = 100:100:1000;

for i = 1:length(Mfs);
    fprintf('Running simulation with %d particles...\n', Mfs(i));
    tstart = tic;
    run_nonmonotonic_example(Mfs(i));
    toc(tstart);
end
