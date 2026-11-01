---
layout: post
title: "Linux Namespaces From Inside the Kernel"
date: 2026-01-06 00:00:00 +0000
description: A deep dive into how Linux namespaces actually work inside the Linux kernel.
tags: linux
---

Happy New Year!

This has been in my books for a while. As much as I want to write regularly, I do not have enough cores in my brain to write fast - and whoever designed the scheduling algorithm in my brain (me) really sucks because it keeps context-switching (I keep getting distracted).

Last November, I had the opportunity of a lifetime to speak in Google Devfest Singapore. I spoke about how containers work under the hood. I introduced the requirements of an online judge and why container-like sandboxes are needed. Then how to use Linux primitives to create a minimal container.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/linux-namespaces-devfest.jpg" alt="Google Devfest Singapore 2025" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/linux-namespaces-devfest-01.jpg" alt="Google Devfest Singapore 2025" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

Well, in that talk, I explained about [**namespaces**](https://man7.org/linux/man-pages/man7/namespaces.7.html) which are a Linux kernel feature that limits what a process can see. This includes [mounts](https://man7.org/linux/man-pages/man7/mount_namespaces.7.html), [users](https://man7.org/linux/man-pages/man7/user_namespaces.7.html), and even [other processes](https://man7.org/linux/man-pages/man7/pid_namespaces.7.html). Then, I became curious about how the kernel is able to isolate the views of different resources among different namespaces. Therefore, the motivation for this blog.

## Namespaces

Namespaces are a feature of the Linux kernel that control the resources that are (or can be) seen by processes. This means that processes in a namespace sees resources differently compared to processes in a different namespace.

As of kernel version 5.6, there are 8 namespace types:

- Mount
- PID
- Network
- IPC
- UTS
- User
- Control group
- Time

Each process belongs to a namespace and can only see resources that are associated with that namespace, for example, a process in a PID namespace can only see processes inside that namespace.

It goes without saying, namespaces are crucial for the modern containers we all know and love.

We can create namespaces using the `unshare` or `clone` system calls. We will talk more about these later on in this blog.

## A Process from the Kernel’s Perspective

To understand how namespaces appear from the kernel’s perspective, we should first take a look at how a process is represented. We can find this by reading the [kernel code](https://elixir.bootlin.com/linux/v6.18.3/source/kernel) (latest version used as of writing is v6.18.3).

It’s a long struct, exactly 850 lines, but you can take a look at it [here](https://elixir.bootlin.com/linux/v6.18.3/source/include/linux/sched.h#L819).

Turns out there’s an interesting field:

```c
struct task_struct
{
    ...
	/* Namespaces: */
	struct nsproxy *nsproxy;
    ...
} __attribute__((aligned(64)));
```

> The kernel doesn’t actually bother with the concept of processes. It treats both processes and threads as tasks.

Each task has a pointer to a `struct nsproxy`.

## A Process’ `nsproxy`

The implementation for `struct nsproxy` can be found [here](https://elixir.bootlin.com/linux/v6.18.3/source/include/linux/nsproxy.h#L32).

```c
/*
 * A structure to contain pointers to all per-process
 * namespaces - fs (mount), uts, network, sysvipc, etc.
 *
 * The pid namespace is an exception -- it's accessed using
 * task_active_pid_ns.  The pid namespace here is the
 * namespace that children will use.
 *
 * 'count' is the number of tasks holding a reference.
 * The count for each namespace, then, will be the number
 * of nsproxies pointing to it, not the number of tasks.
 *
 * The nsproxy is shared by tasks which share all namespaces.
 * As soon as a single namespace is cloned or unshared, the
 * nsproxy is copied.
 */
struct nsproxy {
	refcount_t count;
	struct uts_namespace *uts_ns;
	struct ipc_namespace *ipc_ns;
	struct mnt_namespace *mnt_ns;
	struct pid_namespace *pid_ns_for_children;
	struct net 	     *net_ns;
	struct time_namespace *time_ns;
	struct time_namespace *time_ns_for_children;
	struct cgroup_namespace *cgroup_ns;
};
```

`struct nsproxy` has a pointer to all the respective namespace structs. Since each `task_struct` holds a pointer to an `nsproxy`, every task is associated to a specific set of namespaces.

We’ll see operations involving `nsproxy` [here](https://elixir.bootlin.com/linux/v6.18.3/source/kernel/nsproxy.c).

Tasks can share the same `nsproxy`. Linux uses a copy-on-write for namespace membership. When you `unshare` or `clone`, the kernel copies the `nsproxy` of the parent task and changes only the requested parts.

The initial process (PID 1) has a statically defined `nsproxy` (`init_nsproxy`) which represents the root namespace set. All other tasks either share this `nsproxy` or has a derived copy of it.

```c
struct nsproxy init_nsproxy = {
	.count			= REFCOUNT_INIT(1),
	.uts_ns			= &init_uts_ns,
#if defined(CONFIG_POSIX_MQUEUE) || defined(CONFIG_SYSVIPC)
	.ipc_ns			= &init_ipc_ns,
#endif
	.mnt_ns			= NULL,
	.pid_ns_for_children	= &init_pid_ns,
#ifdef CONFIG_NET
	.net_ns			= &init_net,
#endif
#ifdef CONFIG_CGROUPS
	.cgroup_ns		= &init_cgroup_ns,
#endif
#ifdef CONFIG_TIME_NS
	.time_ns		= &init_time_ns,
	.time_ns_for_children	= &init_time_ns,
#endif
};
```

> If you are wondering why `.mnt_ns = NULL`, it’s because the mount namespace does not exist at the time `init_nsproxy` is defined. Mount namespaces depend on VFS (Virtual File System) which is only initialized after boot. It is assigned later, after the VFS is initialized, the root filesystem is mounted, and the initial task’s `fs_struct` is set up.

## How a Namespace is Created Using `clone`

As was mentioned earlier, you can create namespaces using `clone`. [`clone`](https://man7.org/linux/man-pages/man2/clone.2.html) is a system call that is used to create a child task. It is similar to [`fork`](https://man7.org/linux/man-pages/man2/fork.2.html) but `fork` creates a child with an almost exact copy of the parent’s execution context, `clone` gives the caller fine-grained control over what gets shared and separated. In particular, when using `fork`, the child task inherits the namespaces (by sharing the same `nsproxy`) from the parent. Using `clone` allows the caller to specify the creation of new namespaces.

I used the following code to demonstrate how to create a child process inside a new PID namespace:

```c
void *stack = std::malloc(STACK_SIZE);

if(!stack) {
    perror("Failed to allocate memory for child stack");
    return EXIT_FAILURE;
}

void *stack_top = static_cast<char *>(stack) + STACK_SIZE;

pid_t child_pid = clone(
    child_fn,
    stack_top,
    SIGCHLD | CLONE_NEWPID,
    nullptr
);
```

The `clone` function in C takes in several arguments. The first argument is the function that is to be executed by the child process. The second argument is the child’s stack pointer (pointing to the top of the stack). The third argument is the most important, it holds the flags that can be passed by the caller. The flags dictate what to share and what to create. Namespaces are created by passing such flags (e.g. `CLONE_NEWPID` to create a new PID namespace, `CLONE_NEWNS` to create a new mount namespace, etc.) `clone` then calls the `copy_namespaces` function.

```c
/*
 * called from clone.  This now handles copy for nsproxy and all
 * namespaces therein.
 */
int copy_namespaces(u64 flags, struct task_struct *tsk)
{
	struct nsproxy *old_ns = tsk->nsproxy;
	struct user_namespace *user_ns = task_cred_xxx(tsk, user_ns);
	struct nsproxy *new_ns;

	if (likely(!(flags & (CLONE_NEWNS | CLONE_NEWUTS | CLONE_NEWIPC |
			      CLONE_NEWPID | CLONE_NEWNET |
			      CLONE_NEWCGROUP | CLONE_NEWTIME)))) {
		if ((flags & CLONE_VM) ||
		    likely(old_ns->time_ns_for_children == old_ns->time_ns)) {
			get_nsproxy(old_ns);
			return 0;
		}
	} else if (!ns_capable(user_ns, CAP_SYS_ADMIN))
		return -EPERM;

	/*
	 * CLONE_NEWIPC must detach from the undolist: after switching
	 * to a new ipc namespace, the semaphore arrays from the old
	 * namespace are unreachable.  In clone parlance, CLONE_SYSVSEM
	 * means share undolist with parent, so we must forbid using
	 * it along with CLONE_NEWIPC.
	 */
	if ((flags & (CLONE_NEWIPC | CLONE_SYSVSEM)) ==
		(CLONE_NEWIPC | CLONE_SYSVSEM))
		return -EINVAL;

	new_ns = create_new_namespaces(flags, tsk, user_ns, tsk->fs);
	if (IS_ERR(new_ns))
		return  PTR_ERR(new_ns);

	if ((flags & CLONE_VM) == 0)
		timens_on_fork(new_ns, tsk);

	tsk->nsproxy = new_ns;
	return 0;
}
```

Let’s try to dissect this function.

First, the parent’s `nsproxy` is retrieved along with the user namespace.

```c
struct nsproxy *old_ns = tsk->nsproxy;
struct user_namespace *user_ns = task_cred_xxx(tsk, user_ns);
```

Then it sees whether or not any new namespaces are requested. It checks if the flags contain any of the new namespace flags. If no namespaces are requested, it shares the parent’s `nsproxy`.

```c
if (likely(!(flags & (CLONE_NEWNS | CLONE_NEWUTS | CLONE_NEWIPC |
                CLONE_NEWPID | CLONE_NEWNET |
                CLONE_NEWCGROUP | CLONE_NEWTIME)))) {
    if ((flags & CLONE_VM) ||
        likely(old_ns->time_ns_for_children == old_ns->time_ns)) {
        get_nsproxy(old_ns);
        return 0;
    }
}
```

If new namespaces are to be created, it checks whether the caller has permission. Specifically, the caller must have `CAP_SYS_ADMIN` in its user namespace.

```c
else if (!ns_capable(user_ns, CAP_SYS_ADMIN))
	return -EPERM;
```

Then, it creates the new namespaces.

```c
new_ns = create_new_namespaces(flags, tsk, user_ns, tsk->fs);
tsk->nsproxy = new_ns;
```

In the end, the new child task will have a pointer to `new_ns`, whether it may be the same as its parent or a copy created on write version of it.

> Note: The newly created `task_struct` will by default have its `nsproxy` initialized to point at the parent’s `nsproxy`.

Now, one must wonder what the `create_new_namespaces` function does (I hope you are).

```c
/*
 * Create new nsproxy and all of its the associated namespaces.
 * Return the newly created nsproxy.  Do not attach this to the task,
 * leave it to the caller to do proper locking and attach it to task.
 */
static struct nsproxy *create_new_namespaces(u64 flags,
	struct task_struct *tsk, struct user_namespace *user_ns,
	struct fs_struct *new_fs)
{
	struct nsproxy *new_nsp;
	int err;

	new_nsp = create_nsproxy();
	if (!new_nsp)
		return ERR_PTR(-ENOMEM);

	new_nsp->mnt_ns = copy_mnt_ns(flags, tsk->nsproxy->mnt_ns, user_ns, new_fs);
	if (IS_ERR(new_nsp->mnt_ns)) {
		err = PTR_ERR(new_nsp->mnt_ns);
		goto out_ns;
	}

	new_nsp->uts_ns = copy_utsname(flags, user_ns, tsk->nsproxy->uts_ns);
	if (IS_ERR(new_nsp->uts_ns)) {
		err = PTR_ERR(new_nsp->uts_ns);
		goto out_uts;
	}

	new_nsp->ipc_ns = copy_ipcs(flags, user_ns, tsk->nsproxy->ipc_ns);
	if (IS_ERR(new_nsp->ipc_ns)) {
		err = PTR_ERR(new_nsp->ipc_ns);
		goto out_ipc;
	}

	new_nsp->pid_ns_for_children =
		copy_pid_ns(flags, user_ns, tsk->nsproxy->pid_ns_for_children);
	if (IS_ERR(new_nsp->pid_ns_for_children)) {
		err = PTR_ERR(new_nsp->pid_ns_for_children);
		goto out_pid;
	}

	new_nsp->cgroup_ns = copy_cgroup_ns(flags, user_ns,
					    tsk->nsproxy->cgroup_ns);
	if (IS_ERR(new_nsp->cgroup_ns)) {
		err = PTR_ERR(new_nsp->cgroup_ns);
		goto out_cgroup;
	}

	new_nsp->net_ns = copy_net_ns(flags, user_ns, tsk->nsproxy->net_ns);
	if (IS_ERR(new_nsp->net_ns)) {
		err = PTR_ERR(new_nsp->net_ns);
		goto out_net;
	}

	new_nsp->time_ns_for_children = copy_time_ns(flags, user_ns,
					tsk->nsproxy->time_ns_for_children);
	if (IS_ERR(new_nsp->time_ns_for_children)) {
		err = PTR_ERR(new_nsp->time_ns_for_children);
		goto out_time;
	}
	new_nsp->time_ns = get_time_ns(tsk->nsproxy->time_ns);

	return new_nsp;

out_time:
	put_net(new_nsp->net_ns);
out_net:
	put_cgroup_ns(new_nsp->cgroup_ns);
out_cgroup:
	put_pid_ns(new_nsp->pid_ns_for_children);
out_pid:
	put_ipc_ns(new_nsp->ipc_ns);
out_ipc:
	put_uts_ns(new_nsp->uts_ns);
out_uts:
	put_mnt_ns(new_nsp->mnt_ns);
out_ns:
	kmem_cache_free(nsproxy_cachep, new_nsp);
	return ERR_PTR(err);
}
```

First it initializes an empty `nsproxy`. Then, for each individual namespace (say PID namespace), it checks if the flag has that particular namespace (if `flag & CLONE_NEWPID`). If it does, it creates a brand new namespace, otherwise it will reuse the existing namespace by incrementing the reference count.

Here’s an example for PID namespaces:

```c
struct pid_namespace *copy_pid_ns(u64 flags,
	struct user_namespace *user_ns, struct pid_namespace *old_ns)
{
	if (!(flags & CLONE_NEWPID))
		return get_pid_ns(old_ns);
	if (task_active_pid_ns(current) != old_ns)
		return ERR_PTR(-EINVAL);
	return create_pid_namespace(user_ns, old_ns);
}
```

## Namespaces from the Kernel’s POV

Now that we know how namespaces are created for each task, we can be curious about how namespaces are represented by the kernel, hopefully curiosity doesn’t kill the engineer.

Let’s take a closer look at `struct pid_namespace`.

```c
struct pid_namespace {
	struct idr idr;
	struct rcu_head rcu;
	unsigned int pid_allocated;
	struct task_struct *child_reaper;
	struct kmem_cache *pid_cachep;
	unsigned int level;
	int pid_max;
	struct pid_namespace *parent;
#ifdef CONFIG_BSD_PROCESS_ACCT
	struct fs_pin *bacct;
#endif
	struct user_namespace *user_ns;
	struct ucounts *ucounts;
	int reboot;	/* group exit code if this pidns was rebooted */
	struct ns_common ns;
	struct work_struct	work;
#ifdef CONFIG_SYSCTL
	struct ctl_table_set	set;
	struct ctl_table_header *sysctls;
#if defined(CONFIG_MEMFD_CREATE)
	int memfd_noexec_scope;
#endif
#endif
} __randomize_layout;
```

You know what, we can find more patterns if we use another one. Here’s `struct mnt_namespace`:

```c
struct mnt_namespace {
	struct ns_common	ns;
	struct mount *	root;
	struct {
		struct rb_root	mounts;		 /* Protected by namespace_sem */
		struct rb_node	*mnt_last_node;	 /* last (rightmost) mount in the rbtree */
		struct rb_node	*mnt_first_node; /* first (leftmost) mount in the rbtree */
	};
	struct user_namespace	*user_ns;
	struct ucounts		*ucounts;
	wait_queue_head_t	poll;
	u64			seq_origin; /* Sequence number of origin mount namespace */
	u64 event;
#ifdef CONFIG_FSNOTIFY
	__u32			n_fsnotify_mask;
	struct fsnotify_mark_connector __rcu *n_fsnotify_marks;
#endif
	unsigned int		nr_mounts; /* # of mounts in the namespace */
	unsigned int		pending_mounts;
	refcount_t		passive; /* number references not pinning @mounts */
} __randomize_layout;
```

Namespaces have a generic `ns` field which contains a `struct ns_common`.

```c
struct ns_common {
	u32 ns_type;
	struct dentry *stashed;
	const struct proc_ns_operations *ops;
	unsigned int inum;
	refcount_t __ns_ref; /* do not use directly */
	union {
		struct {
			u64 ns_id;
			struct rb_node ns_tree_node;
			struct list_head ns_list_node;
		};
		struct rcu_head ns_rcu;
	};
};
```

`ns_common` serves as the kernel’s generic “base struct” for namespaces. It lets the kernel identify namespaces, expose namespaces to user space (e.g. via `/proc/<pid>/ns/*`), manage namespace lifetimes (by refcounting and RCU), and track namespaces globally.

Each namespace struct also commonly have a user namespace field, mainly for permissions checking.

## `getpid()` in a PID Namespace

We have seen how namespaces are created and what namespaces look like, but practically, we want to know how is it used. For example, how is it possible that `getpid()` in a user namespace yields a different result if the same process exists in a different namespace.

I made a simple C (actually C++) code that creates a child process in a PID namespace. The parent prints the PID of the child process from the perspective of the parent and the child does the same. The code looks like this:

```cpp
#define _GNU_SOURCE
#include <sched.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <iostream>

constexpr size_t STACK_SIZE = 1024 * 1024;

int child_fn(void *arg) {
    std::cout << "[child] getpid() = " << getpid() << std::endl;
    std::cout << "[child] getppid() = " << getppid() << std::endl;
    return 0;
}

int main(void) {
    std::cout << "[parent] getpid() = " << getpid() << std::endl;
    std::cout << "[parent] getppid() = " << getppid() << std::endl;

    void *stack = std::malloc(STACK_SIZE);

    if(!stack) {
        perror("Failed to allocate memory for child stack");
        return EXIT_FAILURE;
    }

    void *stack_top = static_cast<char *>(stack) + STACK_SIZE;

    pid_t child_pid = clone(
        child_fn,
        stack_top,
        SIGCHLD | CLONE_NEWPID,
        nullptr
    );

    if(child_pid == -1) {
        perror("Failed to create child process");
        std::free(stack);
        return EXIT_FAILURE;
    }

    std::cout << "[parent] Created child process with PID: " << child_pid << std::endl;

    int status = 0;
    waitpid(child_pid, &status, 0);

    std::free(stack);
    return 0;
}
```

```bash
g++ -D_GNU_SOURCE -std=c++23 -Wall -Wextra pidns_getpid.cpp -o pidns_getpid
```

The output is as follows:

```
[parent] getpid() = 387665
[parent] getppid() = 387664
[parent] Created child process with PID: 387666
[child] getpid() = 1
[child] getppid() = 0
```

> `getppid()` gets the PID of the parent.

First, we need to know what `getpid()` does. Internally, `getpid()` calls `task_tgid_vnr()` which is a kernel helper that returns a task’s process ID (TGID). `task_tgid_vnr()` calls `__task_pid_nr_ns()`. For more details see [here](https://elixir.bootlin.com/linux/v6.18.3/source/kernel/pid.c#L510).

> TGID is what the userspace calls the process ID

```c
static inline pid_t task_tgid_vnr(struct task_struct *tsk)
{
	return __task_pid_nr_ns(tsk, PIDTYPE_TGID, NULL);
}
```

```c
pid_t __task_pid_nr_ns(struct task_struct *task, enum pid_type type,
			struct pid_namespace *ns)
{
	pid_t nr = 0;

	rcu_read_lock();
	if (!ns)
		ns = task_active_pid_ns(current);
	if (ns)
		nr = pid_nr_ns(rcu_dereference(*task_pid_ptr(task, type)), ns);
	rcu_read_unlock();

	return nr;
}
```

Since `pid_namespace` is `NULL`, it uses the current PID namespace. Then it calls `pid_nr_ns`.

```c
pid_t pid_nr_ns(struct pid *pid, struct pid_namespace *ns)
{
	struct upid *upid;
	pid_t nr = 0;

	if (pid && ns && ns->level <= pid->level) {
		upid = &pid->numbers[ns->level];
		if (upid->ns == ns)
			nr = upid->nr;
	}
	return nr;
}
```

PID numbers are stored in the task’s `struct pid`.

```c
struct pid {
	refcount_t count;
	unsigned int level;
	spinlock_t lock;
	struct {
		u64 ino;
		struct rb_node pidfs_node;
		struct dentry *stashed;
		struct pidfs_attr *attr;
	};
	/* lists of tasks that use this pid */
	struct hlist_head tasks[PIDTYPE_MAX];
	struct hlist_head inodes;
	/* wait queue for pidfd notifications */
	wait_queue_head_t wait_pidfd;
	struct rcu_head rcu;
	struct upid numbers[];
};
```

`numbers[]` store the PID as seen in different namespaces. PID namespace is hierarchical, it is stored in a flat array of `struct upid` where `numbers[level]` holds the PID of the process as seen in the namespace of depth `level`. So a task that lives in a deep namespace has multiple valid PIDs - one per ancestor namespace.

For example, assume the hierarchy looks like this:

```
init_pid_ns
  └── container_pid_ns
        └── nested_pid_ns
```

The array will be structured like this:

```
pid->numbers[0] → host PID
pid->numbers[1] → container PID
pid->numbers[2] → nested PID
```

Basically, `pid_nr_ns` retrieves the PID of the task associated to the current namespace. A long explanation for a simple conclusion.

## Conclusion

The kernel is namespace-aware. When a function is called to read some namespaced resource (e.g. PIDs, mounts, network interfaces), the kernel determines the active namespace of the caller task and uses the namespace object to view resources. Each namespace maintains its own state, thus just by following pointers, the kernel is able to return different results depending on the caller’s namespace context.

This is my first time doing a deep dive on kernel code. If there are anything that needs clarification or fixing, please do [email](mailto:joshjms1607@gmail.com) me. Many thanks for reading this far!
