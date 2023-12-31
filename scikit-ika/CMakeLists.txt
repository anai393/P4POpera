cmake_minimum_required(VERSION 3.10)

project(skika VERSION 0.0.1)

set(ensemble_sourcefiles
skika/ensemble/adaptive_random_forest.cpp
skika/ensemble/pearl.cpp
skika/ensemble/lru_state.cpp
skika/ensemble/lossy_state_graph.cpp
skika/ensemble/nacre.cpp
)

set(opera_sourcefiles
skika/transfer/opera/opera_wrapper.cpp
skika/transfer/opera/opera.cpp
skika/transfer/opera/phantom_tree.cpp
skika/transfer/opera/random_forest.cpp
)

set(aotradaboost_sourcefiles
skika/transfer/aotradaboost/trans_tree_wrapper.cpp
skika/transfer/aotradaboost/trans_tree.cpp
)

set(ensemble_pybindfiles
skika/ensemble/py_binding.cpp
)

set(opera_pybindfiles
skika/transfer/opera/py_binding.cpp
)

set(aotradaboost_pybindfiles
skika/transfer/aotradaboost/py_binding.cpp
)

set(include_dirs
third_party/streamDM/
)

set(glog "-lglog -lpthread")
set(warning "-Wall -Wextra -Wno-sign-compare -Wno-nonnull -Wno-unused-variable")
set(flags "-fPIC -std=c++14 -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE64 -O3 -DUNIX -g")

add_definitions(${flags})

add_subdirectory(third_party/streamDM)

message("PEARL: turn on Python Wrapper")
add_subdirectory(third_party/pybind11)
pybind11_add_module(ensemble SHARED ${ensemble_sourcefiles} ${ensemble_pybindfiles})
pybind11_add_module(transfer SHARED ${opera_sourcefiles} ${opera_pybindfiles})
pybind11_add_module(aotradaboost SHARED ${aotradaboost_sourcefiles} ${aotradaboost_pybindfiles})

target_link_libraries(ensemble PUBLIC streamDM)
target_include_directories(ensemble PUBLIC ${include_dirs})

target_link_libraries(transfer PUBLIC streamDM)
target_include_directories(transfer PUBLIC ${include_dirs})

target_link_libraries(aotradaboost PUBLIC streamDM)
target_include_directories(aotradaboost PUBLIC ${include_dirs})
