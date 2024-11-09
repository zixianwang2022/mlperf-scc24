cm run script --tags=run-mlperf,inference,_r4.1-dev,_short,_scc24-base    --model=sdxl      --framework=pytorch    --category=datacenter    --scenario=Offline    --execution_mode=test    --device=rocm    --quiet --precision=float16 


# Official Implementation 
# Performance Estimation for Offline Scenario
cm run script --tags=run-mlperf,inference,_find-performance,_r4.1-dev,_short,_scc24-base \
   --model=sdxl \
   --implementation=reference \
   --framework=pytorch \
   --category=datacenter \
   --scenario=Offline \
   --execution_mode=test \
   --device=rocm  \
   --quiet \
    --precision=float16


# Formal run
# Official Implementation 
cm run script --tags=run-mlperf,inference,_r4.1-dev,_short,_scc24-base \
   --model=sdxl \
   --implementation=reference \
   --framework=pytorch \
   --category=datacenter \
   --scenario=Offline \
   --execution_mode=test \
   --device=rocm \
   --quiet --precision=float16



# Custom Implementation 
# Performance Estimation for Offline Scenario
cm run script --tags=run-mlperf,inference,_find-performance,_r4.1-dev,_short,_scc24-base \
   --model=sdxl \
   --framework=pytorch \
   --category=datacenter \
   --scenario=Offline \
   --execution_mode=test \
   --device=rocm  \
   --quiet \
    --precision=float16 \
    --adr.mlperf-implementation.tags=_branch.test,_repo.https://github.com/zixianwang2022/mlperf-scc24 --adr.mlperf-implementation.version=custom 



# Custom Implementation 
# Formal run
cm run script --tags=run-mlperf,inference,_r4.1-dev,_short,_scc24-base \
   --model=sdxl \
   --framework=pytorch \
   --category=datacenter \
   --scenario=Offline \
   --execution_mode=test \
   --device=rocm \
   --quiet --precision=float16 \
   --adr.mlperf-implementation.tags=_branch.test,_repo.https://github.com/zixianwang2022/mlperf-scc24 --adr.mlperf-implementation.version=custom 
   