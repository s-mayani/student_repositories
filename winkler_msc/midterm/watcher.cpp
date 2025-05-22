#include <bitset>
#include <iostream>
#include <thread>
#include <atomic>
#include <sys/inotify.h>
#include <unistd.h>
std::atomic<bool> running, busy;
void trun(){
    while(running){
        for(;!busy;){
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if(!running)return;
        }
        std::system("pdflatex -interaction=nonstopmode midterm.tex");
        busy = false;
    }
}
int main() {
    std::thread ronner;
    running = true;
    busy = false;
    ronner = std::thread(trun);
    // Create an inotify instance
    int inotifyFd = inotify_init();
    if (inotifyFd == -1) {
        std::cerr << "Error initializing inotify." << std::endl;
        return 1;
    }

    // Add a watch for the specified directory or file
    const char* pathToWatch = "midterm.tex";
    int watchDescriptor = inotify_add_watch(inotifyFd, pathToWatch, IN_ALL_EVENTS);

    if (watchDescriptor == -1) {
        std::cerr << "Error adding watch for " << pathToWatch << std::endl;
        close(inotifyFd);
        return 1;
    }

    std::cout << "File watcher is active. Waiting for changes..." << std::endl;

    for (;;) {
        char buffer[1024];
        ssize_t bytesRead = read(inotifyFd, buffer, sizeof(buffer));

        if (bytesRead == -1) {
            std::cerr << "Error reading inotify events." << std::endl;
            break;
        }

        for (char* p = buffer; p < buffer + bytesRead; ) {
            struct inotify_event* event = reinterpret_cast<struct inotify_event*>(p);
            std::cout << "File modified: " << std::bitset<32>(event->mask) << std::endl;
            if (event->mask) {
                busy = true;
                // Add your custom logic here to handle the file change
            }
            

            p += sizeof(struct inotify_event) + event->len;
        }
    }

    // Cleanup
    running = false;
    close(inotifyFd);
    ronner.join();
    return 0;
}
