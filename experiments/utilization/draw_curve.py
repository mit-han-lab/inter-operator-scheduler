import argparse
import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser()
argparser.add_argument('--seq_log_file', type=str, required=True)
argparser.add_argument('--ios_log_file', type=str, required=True)
argparser.add_argument('--running_mean_k', type=int, default=2)
argparser.add_argument('--truncate', type=int, default=0)
args = argparser.parse_args()


def running_mean(x, k): # return sequence with len(x)-k+1 elements
    mx = []
    for i in range(k-1, len(x)):
        mx.append(sum(x[i-k+1:i+1]) / k)
    return mx


def read_log(log_file):
    warps, stamps = [], []
    with open(log_file, 'r') as f:
        event_name = f.readline().split(':')[1]
        for _ in range(3):
            f.readline()
        for line in f.readlines():
            a, b = line.split(' ')
            warps.append(int(a))
            stamps.append(int(b))
    dwarps = [warps[i] - warps[i-1] for i in range(1, len(warps))]
    dwarps = running_mean(dwarps, args.running_mean_k)
    stamps = stamps[args.running_mean_k:]
    n = args.truncate
    if n == 0:
        return event_name, stamps, dwarps
    else:
        return event_name, stamps[:n], dwarps[:n]

readable_event_names = {
    'active_warps_pm': 'ActiveWarps'
}

def main():
    ylen = 100
    event_name = None
    for label, log_file in [('Sequential', args.seq_log_file), ('IOS', args.ios_log_file)]:
        _, _, y = read_log(log_file)
        ylen = min(ylen, len(y))

    for label, log_file in [('Sequential', args.seq_log_file), ('IOS', args.ios_log_file)]:
        ename, _, y = read_log(log_file)
        if event_name is None:
            event_name = ename
        else:
            assert event_name == ename
        y = y[:ylen]
        plt.plot(list(range(1, len(y)+1)), y, label=label)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    event_name = event_name.strip()
    if event_name in readable_event_names:
        event_name = readable_event_names[event_name]
    plt.ylabel(f' {event_name} / Time Stamp')
    plt.xlabel('Time Stamp')
    plt.legend(loc='lower center')
    plt.savefig(f'{event_name}.png')


if __name__ == '__main__':
    main()

