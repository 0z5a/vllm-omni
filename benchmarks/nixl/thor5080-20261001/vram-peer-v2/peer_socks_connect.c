/* Route only the task peer's IPv4 connections through its SSH SOCKS relay. */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>

static pthread_once_t once = PTHREAD_ONCE_INIT;
static int (*native_connect)(int, const struct sockaddr *, socklen_t);
static struct in_addr peer;
static struct sockaddr_in proxy;

static void initialize(void) {
    native_connect = dlsym(RTLD_NEXT, "connect");
    inet_pton(AF_INET, getenv("NIXL_TUNNEL_PEER"), &peer);
    proxy.sin_family = AF_INET;
    inet_pton(AF_INET, getenv("NIXL_TUNNEL_PROXY"), &proxy.sin_addr);
    proxy.sin_port = htons(strtoul(getenv("NIXL_TUNNEL_PORT"), NULL, 10));
}

static int exchange(int fd, unsigned char *data, size_t length, int writing) {
    while (length) {
        ssize_t n = writing ? send(fd, data, length, MSG_NOSIGNAL) : recv(fd, data, length, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { errno = ECONNRESET; return -1; }
        data += n;
        length -= n;
    }
    return 0;
}

int connect(int fd, const struct sockaddr *address, socklen_t length) {
    pthread_once(&once, initialize);
    const struct sockaddr_in *destination = (const struct sockaddr_in *)address;
    if (address->sa_family != AF_INET || destination->sin_addr.s_addr != peer.s_addr)
        return native_connect(fd, address, length);
    int type;
    socklen_t type_length = sizeof(type);
    if (getsockopt(fd, SOL_SOCKET, SO_TYPE, &type, &type_length) || type != SOCK_STREAM)
        return native_connect(fd, address, length);
    int flags = fcntl(fd, F_GETFL);
    if (flags < 0 || fcntl(fd, F_SETFL, flags & ~O_NONBLOCK) < 0) return -1;
    unsigned char hello[] = {5, 1, 0}, answer[10], request[] = {5, 1, 0, 1, 0, 0, 0, 0, 0, 0};
    __builtin_memcpy(request + 4, &destination->sin_addr, 4);
    __builtin_memcpy(request + 8, &destination->sin_port, 2);
    int result = native_connect(fd, (struct sockaddr *)&proxy, sizeof(proxy));
    if (!result) result = exchange(fd, hello, sizeof(hello), 1);
    if (!result) result = exchange(fd, answer, 2, 0);
    if (!result && (answer[0] != 5 || answer[1] != 0)) { errno = EACCES; result = -1; }
    if (!result) result = exchange(fd, request, sizeof(request), 1);
    if (!result) result = exchange(fd, answer, sizeof(answer), 0);
    if (!result && (answer[0] != 5 || answer[1] != 0 || answer[3] != 1)) { errno = EHOSTUNREACH; result = -1; }
    int saved = errno;
    fcntl(fd, F_SETFL, flags);
    errno = saved;
    return result;
}
