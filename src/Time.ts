import { NS } from "@ns";

export class Time {
    private static instance: Time;

    private startTimeMillis = 0;
    private endTimeMillis = 0;
    private sleepTimeMillis = 0;

    public startTime() {
        this.sleepTimeMillis = 0;
        this.startTimeMillis = Date.now();
    }

    public endTime() {
        this.endTimeMillis = Date.now();
    }

    public accumulateSleepTime(time: number) {
        this.sleepTimeMillis += time;
    }

    public getTime(ns: NS) {
        const runningTime = this.endTimeMillis - this.startTimeMillis - this.sleepTimeMillis;
        ns.write("timelog.txt", "Time: " + runningTime + "ms\n", "a");
        return runningTime;
    }

    private constructor() {
        //
    }

    public static getInstance() {
        if (!Time.instance) {
            Time.instance = new Time();
        }
        return Time.instance;
    }
}
