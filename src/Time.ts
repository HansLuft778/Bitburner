import { NS } from "@ns";

export class Time {
    private static instance: Time;

    startTimeValue = 0;
    endTimeValue = 0;

    public startTime() {
        this.startTimeValue = Date.now();
    }

    public endTime() {
        this.endTimeValue = Date.now();
    }

    public substractTime(time: number) {
        this.startTimeValue += time;
    }

    public getTime(ns: NS) {
        ns.write("timelog.txt", "Time: " + (this.endTimeValue - this.startTimeValue) + "ms", "a");
        return this.endTimeValue - this.startTimeValue;
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
