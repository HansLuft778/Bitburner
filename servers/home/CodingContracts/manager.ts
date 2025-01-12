
import { Colors, serverScanner } from "../lib.js";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    const cc = ns.codingcontract;

    const allServers = serverScanner(ns);

    for (const server of allServers) {
        const contracts = ns.ls(server).filter((p) => p.includes("cct"));

        if (contracts.length === 0) continue;

        ns.print(`server ${server} has ${contracts.length} contract(s)`);

        for (const contract of contracts) {
            const contractType = cc.getContractType(contract, server);
            ns.print(`contract ${contract} is of type ${contractType}`);

            let result;
            switch (contractType) {
                case "Find Largest Prime Factor":
                    result = findLargestPrimeFactor(ns, contract, server);
                    break;
                case "Subarray with Maximum Sum":
                    result = subarraywithMaximumSum(ns, contract, server);
                    break;
                case "Total Ways to Sum":
                    result = totalWaysToSum(ns, contract, server);
                    break;
                case "Total Ways to Sum II":
                    result = totalWaysToSumII(ns, contract, server);
                    break;
                case "Spiralize Matrix":
                    result = spiralizeMatrix(ns, contract, server);
                    break;
                case "Array Jumping Game":
                    result = arrayJumpingGame(ns, contract, server);
                    break;
                case "Array Jumping Game II":
                    result = arrayJumpingGameII(ns, contract, server);
                    break;
                case "Merge Overlapping Intervals":
                    result = mergeOverlappingIntervals(ns, contract, server);
                    break;
                case "Generate IP Addresses":
                    result = generateIPAddresses(ns, contract, server);
                    break;
                case "Algorithmic Stock Trader I":
                    result = algorithmicStockTraderI(ns, contract, server);
                    break;
                case "Algorithmic Stock Trader II":
                    result = algorithmicStockTraderII(ns, contract, server);
                    break;
                case "Algorithmic Stock Trader III":
                    result = algorithmicStockTraderIII(ns, contract, server);
                    break;
                case "Algorithmic Stock Trader IV":
                    result = algorithmicStockTraderIV(ns, contract, server);
                    break;
                case "Minimum Path Sum in a Triangle":
                    result = minimumPathSumInATriangle(ns, contract, server);
                    break;
                case "Unique Paths in a Grid I":
                    result = uniquePathsInAGridI(ns, contract, server);
                    break;
                case "Unique Paths in a Grid II":
                    result = uniquePathsInAGridII(ns, contract, server);
                    break;
                case "Shortest Path in a Grid":
                    result = findShortestPath(ns, contract, server);
                    break;
                case "Sanitize Parentheses in Expression":
                    result = sanitizeParenthesesInExpression(ns, contract, server);
                    break;
                case "Find All Valid Math Expressions":
                    result = findAllValidMathExpressions(ns, contract, server);
                    break;
                case "HammingCodes: Integer to Encoded Binary":
                    result = hmmingCodesEncodeIntegerToBinary(ns, contract, server);
                    break;
                case "HammingCodes: Encoded Binary to Integer":
                    result = hmmingCodesEncodeBinaryToInteger(ns, contract, server);
                    break;
                case "Proper 2-Coloring of a Graph":
                    result = proper2ColoringOfAGraph(ns, contract, server);
                    break;
                case "Compression I: RLE Compression":
                    result = compressionI(ns, contract, server);
                    break;
                case "Compression II: LZ Decompression":
                    result = compressionII(ns, contract, server);
                    break;
                case "Compression III: LZ Compression":
                    result = compressionIII(ns, contract, server);
                    break;
                case "Encryption I: Caesar Cipher":
                    result = encryptionI(ns, contract, server);
                    break;
                case "Encryption II: Vigenère Cipher":
                    result = encryptionII(ns, contract, server);
                    break;
                default:
                    throw new Error("unknown contract type: " + contractType);
            }

            if (result === undefined) continue;

            const success = ns.codingcontract.attempt(result, contract, server);
            if (success === "") {
                ns.tprint(Colors.RED + `failed to solve contract ${contract} on server ${server}`);
            }

            ns.print(Colors.GREEN + success);
        }
    }
}

export function findShortestPath(ns: NS, contract: string, server: string) {
    const data: number[][] = ns.codingcontract.getData(contract, server);
    function findWay(position: number[], end: number[], data: number[][]) {
        const queue: number[][][] = [];

        data[position[0]][position[1]] = 1;
        queue.push([position]); // store a path, not just a position

        while (queue.length > 0) {
            const path = queue.shift(); // get the path out of the queue
            if (path === undefined) break;
            const pos = path[path.length - 1]; // ... and then the last position from it
            const direction = [
                [pos[0] + 1, pos[1]],
                [pos[0], pos[1] + 1],
                [pos[0] - 1, pos[1]],
                [pos[0], pos[1] - 1],
            ];

            for (let i = 0; i < direction.length; i++) {
                // Perform this check first:
                if (direction[i][0] == end[0] && direction[i][1] == end[1]) {
                    // return the path that led to the find
                    return path.concat([end]);
                }

                if (
                    direction[i][0] < 0 ||
                    direction[i][0] >= data.length ||
                    direction[i][1] < 0 ||
                    direction[i][1] >= data[0].length ||
                    data[direction[i][0]][direction[i][1]] != 0
                ) {
                    continue;
                }

                data[direction[i][0]][direction[i][1]] = 1;
                // extend and push the path on the queue
                queue.push(path.concat([direction[i]]));
            }
        }
    }

    function annotate(path: number[][]) {
        // Work through each array to see if we can get to Iteration
        let currentPosition = [0, 0];
        let iteration = "";

        // start at the 2nd array
        for (let i = 1; i < path.length; i++) {
            // check each array element to see which one changed
            if (currentPosition[0] < path[i][0]) iteration = iteration + "D";
            if (currentPosition[0] > path[i][0]) iteration = iteration + "U";

            if (currentPosition[1] < path[i][1]) iteration = iteration + "R";
            if (currentPosition[1] > path[i][1]) iteration = iteration + "L";

            currentPosition = path[i];
        }

        return iteration;
    }
    const path = findWay([0, 0], [data.length - 1, data[0].length - 1], data);
    if (path) return annotate(path);
    return "";
}

export function totalWaysToSum(ns: NS, contract: string, server: string) {
    const data: number = ns.codingcontract.getData(contract, server);

    if (typeof data !== "number" || data < 1) {
        ns.tprint(Colors.RED + "invalid data for contract " + contract + " on server " + server);
        return;
    }

    function sumCombinations(target: number, current: number[] = [], start = 1, result: number[][] = []) {
        if (target === 0) {
            result.push(current.slice()); // Push a copy of the current combination to the result
            return;
        }

        for (let i = start; i <= target; i++) {
            current.push(i);
            sumCombinations(target - i, current, i, result);
            current.pop();
        }

        return result;
    }

    const res = sumCombinations(data)?.filter((arr) => arr.length !== 1).length;
    if (res === undefined) {
        ns.tprint(Colors.RED + "failed to solve contract " + contract + " on server " + server);
        return;
    }
    return res;
}

function totalWaysToSumII(ns: NS, contract: string, server: string) {
    const data = ns.codingcontract.getData(contract, server);
    const target: number = data[0];
    const nums: number[] = data[1];

    const dp = Array(target + 1).fill(0);
    dp[0] = 1;

    for (const number of nums) {
        for (let i = number; i < target + 1; i++) {
            dp[i] += dp[i - number];
        }
    }

    const res = dp[target];
    if (res === undefined) {
        ns.tprint(Colors.RED + "failed to solve contract " + contract + " on server " + server);
        return;
    }

    return res;
}

function findAllValidMathExpressions(ns: NS, contract: string, server: string) {
    const data = ns.codingcontract.getData(contract, server);

    function opStr(mu: number, len: number) {
        const ops = ["", "-", "+", "*"];
        const s: string[] = [];
        while (mu >= 4) {
            s.push(ops[mu % 4]);
            mu -= mu % 4;
            mu /= 4;
        }
        s.push(ops[mu]);
        while (s.length < len) {
            s.push(ops[0]);
        }
        return s;
    }
    // ["012345", N], where N is the assertion
    // each gap between two digits can be one of 4 things (blank, -, +, *),
    // so there are 4^d permutations, where d=length-1
    const answers = [];
    const digits = data[0];
    const assertion = data[1];

    const permutations = Math.pow(4, digits.length - 1);
    for (let i = 0; i < permutations; i++) {
        // turn the permutation number into a list of operators
        const ops = opStr(i, digits.length - 1);
        // interleave digits and ops
        let expr = "";
        for (let j = 0; j < ops.length; j++) {
            expr += digits[j] + ops[j];
        }
        expr += digits[ops.length];
        // leading 0s sometimes throw an error about octals
        try {
            if (eval(expr) == assertion) {
                answers.push(expr);
            }
        } catch (e) {
            ns.tprint("Error while solving findAllValidMathExpressions: " + e);
        }
    }

    return answers;
}

function algorithmicStockTraderI(ns: NS, contract: string, server: string) {
    const data = ns.codingcontract.getData(contract, server);
    let maxCur = 0;
    let maxSoFar = 0;
    for (let i = 1; i < data.length; ++i) {
        maxCur = Math.max(0, (maxCur += data[i] - data[i - 1]));
        maxSoFar = Math.max(maxCur, maxSoFar);
    }

    return maxSoFar.toString();
}

function algorithmicStockTraderII(ns: NS, contract: string, server: string) {
    const stockPrice: number[] = ns.codingcontract.getData(contract, server);

    stockPrice.push(0);

    let profit = 0;
    for (let i = 0; i < stockPrice.length; i++) {
        if (stockPrice[i] < stockPrice[i + 1]) {
            profit += stockPrice[i + 1] - stockPrice[i];
        }
    }

    profit = Math.max(profit, 0);

    return profit;
}

export function algorithmicStockTraderIII(ns: NS, contract: string, server: string) {
    const stockPrice: number[] = ns.codingcontract.getData(contract, server);

    let hold1 = Number.MIN_SAFE_INTEGER;
    let hold2 = Number.MIN_SAFE_INTEGER;
    let release1 = 0;
    let release2 = 0;
    for (const price of stockPrice) {
        release2 = Math.max(release2, hold2 + price);
        hold2 = Math.max(hold2, release1 - price);
        release1 = Math.max(release1, hold1 + price);
        hold1 = Math.max(hold1, price * -1);
    }

    return release2;
}

function algorithmicStockTraderIV(ns: NS, contract: string, server: string) {
    const data = ns.codingcontract.getData(contract, server);
    const k: number = data[0];
    const prices: number[] = data[1];

    const len = prices.length;

    const hold: number[] = [];
    const rele: number[] = [];
    hold.length = k + 1;
    rele.length = k + 1;
    for (let i = 0; i <= k; ++i) {
        hold[i] = Number.MIN_SAFE_INTEGER;
        rele[i] = 0;
    }

    let cur: number;
    for (let i = 0; i < len; ++i) {
        cur = prices[i];
        for (let j = k; j > 0; --j) {
            rele[j] = Math.max(rele[j], hold[j] + cur);
            hold[j] = Math.max(hold[j], rele[j - 1] - cur);
        }
    }

    return rele[k];
}

function minimumPathSumInATriangle(ns: NS, contract: string, server: string) {
    const triangle: number[][] = ns.codingcontract.getData(contract, server);
    ns.print(triangle);

    const n: number = triangle.length;
    const dp: number[] = triangle[n - 1].slice();
    for (let i = n - 2; i > -1; --i) {
        for (let j = 0; j < triangle[i].length; ++j) {
            dp[j] = Math.min(dp[j], dp[j + 1]) + triangle[i][j];
        }
    }

    return dp[0];
}

function subarraywithMaximumSum(ns: NS, contract: string, server: string) {
    const data: number[] = ns.codingcontract.getData(contract, server);

    let maxSum = -Infinity;
    let currentSum = 0;

    for (let i = 0; i < data.length; i++) {
        currentSum = Math.max(data[i], currentSum + data[i]);
        maxSum = Math.max(maxSum, currentSum);
    }

    return maxSum;
}

function uniquePathsInAGridI(ns: NS, contract: string, server: string) {
    const data = ns.codingcontract.getData(contract, server);
    const n: number = data[0]; // Number of rows
    const m: number = data[1]; // Number of columns
    const currentRow: number[] = [];
    currentRow.length = n;

    for (let i = 0; i < n; i++) {
        currentRow[i] = 1;
    }
    for (let row = 1; row < m; row++) {
        for (let i = 1; i < n; i++) {
            currentRow[i] += currentRow[i - 1];
        }
    }

    return currentRow[n - 1];
}

function uniquePathsInAGridII(ns: NS, contract: string, server: string) {
    const data = ns.codingcontract.getData(contract, server);
    const obstacleGrid: number[][] = [];
    obstacleGrid.length = data.length;
    for (let i = 0; i < obstacleGrid.length; ++i) {
        obstacleGrid[i] = data[i].slice();
    }

    for (let i = 0; i < obstacleGrid.length; i++) {
        for (let j = 0; j < obstacleGrid[0].length; j++) {
            if (obstacleGrid[i][j] == 1) {
                obstacleGrid[i][j] = 0;
            } else if (i == 0 && j == 0) {
                obstacleGrid[0][0] = 1;
            } else {
                obstacleGrid[i][j] = (i > 0 ? obstacleGrid[i - 1][j] : 0) + (j > 0 ? obstacleGrid[i][j - 1] : 0);
            }
        }
    }

    return obstacleGrid[obstacleGrid.length - 1][obstacleGrid[0].length - 1];
}

function generateIPAddresses(ns: NS, contract: string, server: string) {
    const data = ns.codingcontract.getData(contract, server);
    const ret: string[] = [];
    for (let a = 1; a <= 3; ++a) {
        for (let b = 1; b <= 3; ++b) {
            for (let c = 1; c <= 3; ++c) {
                for (let d = 1; d <= 3; ++d) {
                    if (a + b + c + d === data.length) {
                        const A = parseInt(data.substring(0, a), 10);
                        const B = parseInt(data.substring(a, a + b), 10);
                        const C = parseInt(data.substring(a + b, a + b + c), 10);
                        const D = parseInt(data.substring(a + b + c, a + b + c + d), 10);
                        if (A <= 255 && B <= 255 && C <= 255 && D <= 255) {
                            const ip: string = [
                                A.toString(),
                                ".",
                                B.toString(),
                                ".",
                                C.toString(),
                                ".",
                                D.toString(),
                            ].join("");
                            if (ip.length === data.length + 3) {
                                ret.push(ip);
                            }
                        }
                    }
                }
            }
        }
    }

    return ret;
}

function encryptionI(ns: NS, contract: string, server: string) {
    const data = ns.codingcontract.getData(contract, server);

    // data = [plaintext, shift value]
    // build char array, shifting via map and join to final results
    const cipher = [...data[0]]
        .map((a) => (a === " " ? a : String.fromCharCode(((a.charCodeAt(0) - 65 - data[1] + 26) % 26) + 65)))
        .join("");
    return cipher;
}

function encryptionII(ns: NS, contract: string, server: string) {
    const data = ns.codingcontract.getData(contract, server);
    // build char array, shifting via map and corresponding keyword letter and join to final results
    const cipher = [...data[0]]
        .map((a, i) => {
            return a === " "
                ? a
                : String.fromCharCode(((a.charCodeAt(0) - 2 * 65 + data[1].charCodeAt(i % data[1].length)) % 26) + 65);
        })
        .join("");
    return cipher;
}

function sanitizeParenthesesInExpression(ns: NS, contract: string, server: string) {
    const data = ns.codingcontract.getData(contract, server);
    if (typeof data !== "string") throw new Error("solver expected string");
    let left = 0;
    let right = 0;
    const res: string[] = [];

    for (let i = 0; i < data.length; ++i) {
        if (data[i] === "(") {
            ++left;
        } else if (data[i] === ")") {
            left > 0 ? --left : ++right;
        }
    }

    function dfs(
        pair: number,
        index: number,
        left: number,
        right: number,
        s: string,
        solution: string,
        res: string[],
    ): void {
        if (s.length === index) {
            if (left === 0 && right === 0 && pair === 0) {
                for (let i = 0; i < res.length; i++) {
                    if (res[i] === solution) {
                        return;
                    }
                }
                res.push(solution);
            }
            return;
        }

        if (s[index] === "(") {
            if (left > 0) {
                dfs(pair, index + 1, left - 1, right, s, solution, res);
            }
            dfs(pair + 1, index + 1, left, right, s, solution + s[index], res);
        } else if (s[index] === ")") {
            if (right > 0) dfs(pair, index + 1, left, right - 1, s, solution, res);
            if (pair > 0) dfs(pair - 1, index + 1, left, right, s, solution + s[index], res);
        } else {
            dfs(pair, index + 1, left, right, s, solution + s[index], res);
        }
    }

    dfs(0, 0, left, right, data, "", res);

    return res;
}

function compressionI(ns: NS, contract: string, server: string) {
    const data = ns.codingcontract.getData(contract, server);
    if (typeof data !== "string") throw new Error("solver expected string");

    let pos = 0;
    let i = 1;
    const length = data.length;
    let compression = "";

    // go through each letter
    while (pos < length) {
        // Check each letter to see if it matches the next
        if (data.charAt(pos) == data.charAt(pos + 1)) {
            // add a position increase for that letter
            i++;
        } else {
            // check if there are more than 10 iterations
            if (i > 9) {
                // How many 9's
                const split = Math.floor(i / 9);
                for (let n = 0; n < split; n++) {
                    compression += "9" + data.charAt(pos);
                }
                //Add the remaining number left
                compression += i - split * 9 + data.charAt(pos);
            } else {
                // if the next letter doesn't match then we need to write out to the compression string
                compression += i + data.charAt(pos);
            }
            i = 1;
        }
        pos++;
    }
    return compression;
}

function compressionII(ns: NS, contract: string, server: string) {
    const compr = ns.codingcontract.getData(contract, server);

    let plain = "";

    for (let i = 0; i < compr.length; ) {
        const literal_length = compr.charCodeAt(i) - 0x30;

        if (literal_length < 0 || literal_length > 9 || i + 1 + literal_length > compr.length) {
            return null;
        }

        plain += compr.substring(i + 1, i + 1 + literal_length);
        i += 1 + literal_length;

        if (i >= compr.length) {
            break;
        }
        const backref_length = compr.charCodeAt(i) - 0x30;

        if (backref_length < 0 || backref_length > 9) {
            return null;
        } else if (backref_length === 0) {
            ++i;
        } else {
            if (i + 1 >= compr.length) {
                return null;
            }

            const backref_offset = compr.charCodeAt(i + 1) - 0x30;
            if ((backref_length > 0 && (backref_offset < 1 || backref_offset > 9)) || backref_offset > plain.length) {
                return null;
            }

            for (let j = 0; j < backref_length; ++j) {
                plain += plain[plain.length - backref_offset];
            }

            i += 2;
        }
    }

    return plain;
}

function compressionIII(ns: NS, contract: string, server: string) {
    const plain = ns.codingcontract.getData(contract, server);
    if (typeof plain !== "string") throw new Error("solver expected string");

    // for state[i][j]:
    //      if i is 0, we're adding a literal of length j
    //      else, we're adding a backreference of offset i and length j
    let cur_state: (string | null)[][] = Array.from(Array(10), () => Array(10).fill(null));
    let new_state: (string | null)[][] = Array.from(Array(10), () => Array(10));

    function set(state: (string | null)[][], i: number, j: number, str: string): void {
        const current = state[i][j];
        if (current == null || str.length < current.length) {
            state[i][j] = str;
        } else if (str.length === current.length && Math.random() < 0.5) {
            // if two strings are the same length, pick randomly so that
            // we generate more possible inputs to Compression II
            state[i][j] = str;
        }
    }

    // initial state is a literal of length 1
    cur_state[0][1] = "";

    for (let i = 1; i < plain.length; ++i) {
        for (const row of new_state) {
            row.fill(null);
        }
        const c = plain[i];

        // handle literals
        for (let length = 1; length <= 9; ++length) {
            const string = cur_state[0][length];
            if (string == null) {
                continue;
            }

            if (length < 9) {
                // extend current literal
                set(new_state, 0, length + 1, string);
            } else {
                // start new literal
                set(new_state, 0, 1, string + "9" + plain.substring(i - 9, i) + "0");
            }

            for (let offset = 1; offset <= Math.min(9, i); ++offset) {
                if (plain[i - offset] === c) {
                    // start new backreference
                    set(new_state, offset, 1, string + String(length) + plain.substring(i - length, i));
                }
            }
        }

        // handle backreferences
        for (let offset = 1; offset <= 9; ++offset) {
            for (let length = 1; length <= 9; ++length) {
                const string = cur_state[offset][length];
                if (string == null) {
                    continue;
                }

                if (plain[i - offset] === c) {
                    if (length < 9) {
                        // extend current backreference
                        set(new_state, offset, length + 1, string);
                    } else {
                        // start new backreference
                        set(new_state, offset, 1, string + "9" + String(offset) + "0");
                    }
                }

                // start new literal
                set(new_state, 0, 1, string + String(length) + String(offset));

                // end current backreference and start new backreference
                for (let new_offset = 1; new_offset <= Math.min(9, i); ++new_offset) {
                    if (plain[i - new_offset] === c) {
                        set(new_state, new_offset, 1, string + String(length) + String(offset) + "0");
                    }
                }
            }
        }

        const tmp_state = new_state;
        new_state = cur_state;
        cur_state = tmp_state;
    }

    let result = null;

    for (let len = 1; len <= 9; ++len) {
        let string = cur_state[0][len];
        if (string == null) {
            continue;
        }

        string += String(len) + plain.substring(plain.length - len, plain.length);
        if (result == null || string.length < result.length) {
            result = string;
        } else if (string.length == result.length && Math.random() < 0.5) {
            result = string;
        }
    }

    for (let offset = 1; offset <= 9; ++offset) {
        for (let len = 1; len <= 9; ++len) {
            let string = cur_state[offset][len];
            if (string == null) {
                continue;
            }

            string += String(len) + "" + String(offset);
            if (result == null || string.length < result.length) {
                result = string;
            } else if (string.length == result.length && Math.random() < 0.5) {
                result = string;
            }
        }
    }

    if (result == null) throw new Error("no result found");
    return result;
}

/* eslint-disable @typescript-eslint/no-explicit-any */

function hmmingCodesEncodeIntegerToBinary(ns: NS, contract: string, server: string) {
    const data = ns.codingcontract.getData(contract, server);
    if (typeof data !== "number") throw new Error("solver expected number");

    const enc: number[] = [0];
    const data_bits: any[] = data.toString(2).split("").reverse();

    data_bits.forEach((e, i, a) => {
        a[i] = parseInt(e);
    });

    let k = data_bits.length;

    /* NOTE: writing the data like this flips the endianness, this is what the
     * original implementation by Hedrauta did so I'm keeping it like it was. */
    for (let i = 1; k > 0; i++) {
        if ((i & (i - 1)) != 0) {
            enc[i] = data_bits[--k];
        } else {
            enc[i] = 0;
        }
    }

    let parity: any = 0;
    /* Figure out the subsection parities */
    for (let i = 0; i < enc.length; i++) {
        if (enc[i]) {
            parity ^= i;
        }
    }

    parity = parity.toString(2).split("").reverse();
    parity.forEach((e: any, i: any, a: any) => {
        a[i] = parseInt(e);
    });

    /* Set the parity bits accordingly */
    for (let i = 0; i < parity.length; i++) {
        enc[2 ** i] = parity[i] ? 1 : 0;
    }

    parity = 0;
    /* Figure out the overall parity for the entire block */
    for (let i = 0; i < enc.length; i++) {
        if (enc[i]) {
            parity++;
        }
    }

    /* Finally set the overall parity bit */
    enc[0] = parity % 2 == 0 ? 0 : 1;

    return enc.join("");
}
/* eslint-enable @typescript-eslint/no-explicit-any */

function hmmingCodesEncodeBinaryToInteger(ns: NS, contract: string, server: string) {
    const data = ns.codingcontract.getData(contract, server);
    if (typeof data !== "string") throw new Error("solver expected string");

    let err = 0;
    const bits: number[] = [];

    /* TODO why not just work with an array of digits from the start? */
    for (const i in data.split("")) {
        const bit = parseInt(data[i]);
        bits[i] = bit;

        if (bit) {
            err ^= +i;
        }
    }

    /* If err != 0 then it spells out the index of the bit that was flipped */
    if (err) {
        /* Flip to correct */
        bits[err] = bits[err] ? 0 : 1;
    }

    /* Now we have to read the message, bit 0 is unused (it's the overall parity bit
     * which we don't care about). Each bit at an index that is a power of 2 is
     * a parity bit and not part of the actual message. */

    let ans = "";

    for (let i = 1; i < bits.length; i++) {
        /* i is not a power of two so it's not a parity bit */
        if ((i & (i - 1)) != 0) {
            ans += bits[i];
        }
    }

    /* TODO to avoid ambiguity about endianness why not let the player return the extracted (and corrected)
     * data bits, rather than guessing at how to convert it to a decimal string? */
    return parseInt(ans, 2);
}

function arrayJumpingGame(ns: NS, contract: string, server: string) {
    const data = ns.codingcontract.getData(contract, server);
    const n: number = data.length;
    let i = 0;
    for (let reach = 0; i < n && i <= reach; ++i) {
        reach = Math.max(i + data[i], reach);
    }
    const solution: boolean = i === n;
    return solution ? "1" : "0";
}

function arrayJumpingGameII(ns: NS, contract: string, server: string) {
    const data = ns.codingcontract.getData(contract, server);

    const n: number = data.length;
    let reach = 0;
    let jumps = 0;
    let lastJump = -1;
    while (reach < n - 1) {
        let jumpedFrom = -1;
        for (let i = reach; i > lastJump; i--) {
            if (i + data[i] > reach) {
                reach = i + data[i];
                jumpedFrom = i;
            }
        }
        if (jumpedFrom === -1) {
            jumps = 0;
            break;
        }
        lastJump = jumpedFrom;
        jumps++;
    }
    return jumps;
}

function convert2DArrayToString(arr: unknown[][]): string {
    const components: string[] = [];
    arr.forEach((e: unknown) => {
        let s = String(e);
        s = ["[", s, "]"].join("");
        components.push(s);
    });

    return components.join(",").replace(/\s/g, "");
}

function mergeOverlappingIntervals(ns: NS, contract: string, server: string) {
    const data = ns.codingcontract.getData(contract, server);

    const intervals: number[][] = data.slice();
    intervals.sort((a: number[], b: number[]) => {
        return a[0] - b[0];
    });

    const result: number[][] = [];
    let start: number = intervals[0][0];
    let end: number = intervals[0][1];
    for (const interval of intervals) {
        if (interval[0] <= end) {
            end = Math.max(end, interval[1]);
        } else {
            result.push([start, end]);
            start = interval[0];
            end = interval[1];
        }
    }
    result.push([start, end]);

    const sanitizedResult: string = convert2DArrayToString(result);
    return sanitizedResult;
}

function proper2ColoringOfAGraph(ns: NS, contract: string, server: string) {
    const data: [number, [number, number][]] = ns.codingcontract.getData(contract, server);

    //Helper function to get neighbourhood of a vertex
    function neighbourhood(vertex: unknown) {
        const adjLeft = data[1].filter(([a]) => a === vertex).map(([, b]) => b);
        const adjRight = data[1].filter(([, b]) => b === vertex).map(([a]) => a);
        return adjLeft.concat(adjRight);
    }

    //Verify that there is no solution by attempting to create a proper 2-coloring.
    const coloring = Array(data[0]).fill(undefined);
    while (coloring.some((val) => val === undefined)) {
        //Color a vertex in the graph
        const initialVertex = coloring.findIndex((val) => val === undefined);
        coloring[initialVertex] = 0;
        const frontier = [initialVertex];

        //Propagate the coloring throughout the component containing v greedily
        while (frontier.length > 0) {
            const v = frontier.pop() || 0;
            const neighbors = neighbourhood(v);

            //For each vertex u adjacent to v
            for (const id in neighbors) {
                const u = neighbors[id];

                //Set the color of u to the opposite of v's color if it is new,
                //then add u to the frontier to continue the algorithm.
                if (coloring[u] === undefined) {
                    if (coloring[v] === 0) coloring[u] = 1;
                    else coloring[u] = 0;

                    frontier.push(u);
                }

                //Assert u,v do not have the same color
                else if (coloring[u] === coloring[v]) {
                    //If u,v do have the same color, no proper 2-coloring exists
                    return "";
                }
            }
        }
    }

    //If this code is reached, there exists a proper 2-coloring of the input
    return coloring;
}

function findLargestPrimeFactor(ns: NS, contract: string, server: string) {
    const data = ns.codingcontract.getData(contract, server);
    if (typeof data !== "number") throw new Error("solver expected number");

    let fac = 2;
    let n: number = data;
    while (n > (fac - 1) * (fac - 1)) {
        while (n % fac === 0) {
            n = Math.round(n / fac);
        }
        ++fac;
    }

    return n === 1 ? fac - 1 : n;
}

function spiralizeMatrix(ns: NS, contract: string, server: string) {
    const data: number[][] = ns.codingcontract.getData(contract, server);

    const spiral: number[] = [];
    const m: number = data.length;
    const n: number = data[0].length;
    let u = 0;
    let d: number = m - 1;
    let l = 0;
    let r: number = n - 1;
    let k = 0;
    let done = false;
    while (!done) {
        // Up
        for (let col: number = l; col <= r; col++) {
            spiral[k] = data[u][col];
            ++k;
        }
        if (++u > d) {
            done = true;
            continue;
        }

        // Right
        for (let row: number = u; row <= d; row++) {
            spiral[k] = data[row][r];
            ++k;
        }
        if (--r < l) {
            done = true;
            continue;
        }

        // Down
        for (let col: number = r; col >= l; col--) {
            spiral[k] = data[d][col];
            ++k;
        }
        if (--d < u) {
            done = true;
            continue;
        }

        // Left
        for (let row: number = d; row >= u; row--) {
            spiral[k] = data[row][l];
            ++k;
        }
        if (++l > r) {
            done = true;
            continue;
        }
    }

    return spiral;
}
