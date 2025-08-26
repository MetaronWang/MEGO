
add_executable(COMIMP ${SOURCE_FILES} ${HEADER_FILES})

target_link_libraries(COMIMP
        PUBLIC
        Boost::system
        Boost::filesystem
        Boost::thread
        Boost::chrono
        Boost::log
        Boost::random
)