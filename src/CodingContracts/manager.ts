import { NS } from "@ns";
import { Colors, serverScanner } from "@/lib";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    const cc = ns.codingcontract;

    const allServers = serverScanner(ns);

    for (const server of allServers) {
        const contracts = ns.ls(server).filter((p) => p.includes("cct"));

        if (contracts.length === 0) continue;

        ns.print(`server ${server} has ${contracts.length} contracts`);

        for (const contract of contracts) {
            const contractType = cc.getContractType(contract, server);
            ns.print(`contract ${contract} is of type ${contractType}`);

            let result;
            switch (contractType) {
                case "Find Largest Prime Factor":
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
                    break;
                case "Array Jumping Game":
                    break;
                case "Array Jumping Game II":
                    break;
                case "Merge Overlapping Intervals":
                    break;
                case "Generate IP Addresses":
                    result = generateIPAddresses(ns, contract, server);
                    break;
                case "Algorithmic Stock Trader I":
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
                    break;
                case "Find All Valid Math Expressions":
                    result = findAllValidMathExpressions(ns, contract, server);
                    break;
                case "HammingCodes: Integer to Encoded Binary":
                    break;
                case "HammingCodes: Encoded Binary to Integer":
                    break;
                case "Proper 2-Coloring of a Graph":
                    break;
                case "Compression I: RLE Compression":
                    break;
                case "Compression II: LZ Decompression":
                    break;
                case "Compression III: LZ Compression":
                    break;
                case "Encryption I: Caesar Cipher":
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
                throw new Error("failed to solve contract");
            }

            ns.print(Colors.GREEN + success);
        }
    }
}

export function findShortestPath(ns: NS, contract: string, server: string) {
    const data: number[][] = ns.codingcontract.getData(contract, server);

    if (data.length === 0 || data[0][0] === 1 || data[data.length - 1][data[0].length - 1] === 1) {
        ns.print("no solution found");
        return;
    }

    const visited: boolean[][] = Array.from({ length: data.length }, () => Array(data[0].length).fill(false));
    const moves: string[] = [""];
    const min = { length: Number.MAX_SAFE_INTEGER };
    const currentPath: string[] = [];

    const startX = 0;
    const startY = 0;
    const endX = data.length - 1;
    const endY = data[0].length - 1;

    const directions = {
        R: [0, 1],
        L: [0, -1],
        D: [1, 0],
        U: [-1, 0],
    };

    function findPath(
        grid: number[][],
        visited: boolean[][],
        i: number,
        j: number,
        x: number,
        y: number,
        moves: string[],
        min: { length: number },
        currentPath: string[],
        directions: Record<string, number[]>,
    ) {
        if (i == x && j == y) {
            if (currentPath.length < min.length) {
                min.length = currentPath.length;
                moves[0] = currentPath.join("");
            }
            return;
        }

        visited[i][j] = true;

        for (const dir in directions) {
            const [row, col] = directions[dir];

            if (isValid(grid, visited, i + row, j + col)) {
                currentPath.push(dir);
                findPath(grid, visited, i + row, j + col, x, y, moves, min, currentPath, directions);
                currentPath.pop();
            }
        }

        visited[i][j] = false;
    }

    function isValid(grid: number[][], visited: boolean[][], row: number, col: number): boolean {
        return (
            row >= 0 && row < grid.length && col >= 0 && col < grid[0].length && !visited[row][col] && !grid[row][col]
        );
    }

    findPath(data, visited, startX, startY, endX, endY, moves, min, currentPath, directions);

    return moves[0];
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

    let pathSum = 0;
    pathSum += triangle[0][0];
    let currentCol = 0;
    for (let i = 0; i < triangle.length; i++) {
        if (triangle[i + 1] === undefined) break;
        const possibleMoves = [triangle[i + 1][currentCol], triangle[i + 1][currentCol + 1]];

        if (possibleMoves[0] > possibleMoves[1]) currentCol = currentCol + 1;
        const move = Math.min(...possibleMoves);
        pathSum += move;
    }

    return pathSum;
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
