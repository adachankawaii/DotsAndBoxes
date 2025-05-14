#include <bits/stdc++.h>

using namespace std;

int dx[4] = {0, -1, 1, 0};
int dy[4] = {-1, 0, 0, 1};

mt19937_64 rnd(chrono::steady_clock::now().time_since_epoch().count());

long long abs_(long long x)
{
    if (x < 0)
        x = -x;
    return x;
}

long long rng(long long l, long long r)
{
    return l + abs_((long long)(rnd())) % (r - l + 1);
}

long long binpow(long long n, long long k)
{
    if (k == 0)
        return 1;
    long long p = binpow(n, k / 2);
    p *= p;
    if (k % 2)
        p *= n;
    return p;
}

string format(int x)
{
    string c = "";
    while (x > 0)
    {
        c = char((x % 10) + 48) + c;
        x /= 10;
    }
    if (c.size() == 1)
        c = "0" + c;
    else if (c.size() == 0)
        c = "00";
    return c;
}

// Old interface
struct OldInterface
{
    int sq_size;
    vector<vector<int>> horz;
    vector<vector<int>> vert;

    // Intital
    void set_size(int cur_size)
    {
        sq_size = cur_size;

        horz.resize(sq_size + 2);

        for (int i = 0; i <= sq_size; i++)
            horz[i].resize(sq_size + 1);

        for (int i = 1; i <= sq_size + 1; i++)
            for (int j = 1; j <= sq_size; j++)
                horz[i][j] = 0;

        vert.resize(sq_size + 1);

        for (int i = 0; i <= sq_size; i++)
            vert[i].resize(sq_size + 2);

        for (int i = 1; i <= sq_size; i++)
            for (int j = 1; j <= sq_size + 1; j++)
                vert[i][j] = 0;
    }
};

// Interface for Genetic Algorithm
struct GeneticInterface
{
    int sq_size;
    int p1_score, p2_score;
    int move_count;
    long long A_chr; // chromosome
    vector<vector<int>> box, vis;
    vector<vector<char>> dis;
    vector<bool> valid_move;
    int long_chain, short_chain, double_cross, box3, box4;

    // each box will have a number of 4 bits, which represents up, down, left and right respectively

    // Intital
    void set_size(int cur_size)
    {
        sq_size = cur_size;

        p1_score = 0, p2_score = 0;

        box.resize(sq_size + 1);

        for (int i = 0; i <= sq_size; i++)
            box[i].resize(sq_size + 1);

        dis.resize(sq_size + 1);

        for (int i = 0; i <= sq_size; i++)
            dis[i].resize(sq_size + 1);

        vis.resize(sq_size + 1);
        for (int i = 0; i <= sq_size; i++)
            vis[i].resize(sq_size + 1);

        for (int i = 1; i <= sq_size; i++)
            for (int j = 1; j <= sq_size; j++)
                box[i][j] = 0, dis[i][j] = ' ', vis[i][j] = 0;

        valid_move.clear();
        valid_move.resize(2 * (sq_size) * (sq_size + 1));
        for (int i = 0; i < 2 * (sq_size) * (sq_size + 1); i++)
            valid_move[i] = 0;
        move_count = 1;
        A_chr = rng(0, (1LL << 60) - 1);
    }

    bool check_if_filled(int i, int j) // check if box (i, j) is filled
    {
        return box[i][j] == 15;
    }

    // Convert to old interface
    OldInterface OldIntConvert()
    {
        OldInterface resInt;

        resInt.set_size(sq_size);

        // Horizontal edges

        // first row, up edge
        for (int i = 1; i <= sq_size; i++)
            if ((box[1][i]) & 1)
                resInt.horz[1][i] = 1;

        // every row, down edge
        for (int i = 1; i <= sq_size; i++)
            for (int j = 1; j <= sq_size; j++)
                if ((box[i][j] >> 1) & 1)
                    resInt.horz[i + 1][j] = 1;

        // Vertical edges

        // first column, left edge
        for (int i = 1; i <= sq_size; i++)
            if ((box[i][1] >> 2) & 1)
                resInt.vert[i][1] = 1;

        // every column, right edge
        for (int i = 1; i <= sq_size; i++)
            for (int j = 1; j <= sq_size; j++)
                if ((box[i][j] >> 3) & 1)
                    resInt.vert[i][j + 1] = 1;

        return resInt;
    }

    /*
    Display:
    Edges are represented by "  "/"--" if horizontal (for box debug), and " "/"-" if vertical.
    Corners are represented by o.
    Boxes are represented by it's freedom degree.
    */

    /*
    Move:
    Define n as tgenes size of tgenes board.
    Tgenesre will be a total of 2 * n * (n + 1) valid moves.

    We will define tgenes order of tgenes moves like this:
    [0; n*n-1]: up edge, every box
    [n*n, 2*n*n-1]: left edge, every box
    [2*n*n, 2*n*n+n-1]: down edge, last row boxes
    [2*n*n+n, 2*n*(n+1)-1]: right edge, last column boxes.

    For example, if n = 3, respective ranges would be [0, 8]; [9, 17]; [18, 20]; [21, 23].
    */

    void move(int id)
    {
        valid_move[id] = 1;

        int n = sq_size;

        // up edge
        if (id < n * n)
        {
            // find box position
            int x = id / n + 1, y = id % n + 1;

            box[x][y] |= 1;
            if (x > 1)
                box[x - 1][y] |= 2;
        }

        // left edge
        else if (id < 2 * n * n)
        {
            // find box position
            id -= n * n;
            int x = id / n + 1, y = id % n + 1;

            box[x][y] |= 4;
            if (y > 1)
                box[x][y - 1] |= 8;
        }

        // down edge
        else if (id < 2 * n * n + n)
        {
            // find box position
            id -= 2 * n * n;
            int y = id + 1;

            box[n][y] |= 2;
        }

        // right edge
        else
        {
            // find box postiton
            id -= 2 * n * n + n;
            int x = id + 1;
            box[x][n] |= 8;
        }
    }

    // undo move
    void unmove(int id)
    {
        valid_move[id] = 0;

        int n = sq_size;

        // up edge
        if (id < n * n)
        {
            // find box position
            int x = id / n + 1, y = id % n + 1;

            box[x][y] ^= 1;
            if (x > 1)
                box[x - 1][y] ^= 2;
        }

        // left edge
        else if (id < 2 * n * n)
        {
            // find box position
            id -= n * n;
            int x = id / n + 1, y = id % n + 1;

            box[x][y] ^= 4;
            if (y > 1)
                box[x][y - 1] ^= 8;
        }

        // down edge
        else if (id < 2 * n * n + n)
        {
            // find box position
            id -= 2 * n * n;
            int y = id + 1;

            box[n][y] ^= 2;
        }

        // right edge
        else
        {
            // find box postiton
            id -= 2 * n * n + n;
            int x = id + 1;
            box[x][n] ^= 8;
        }
    }

    // check if box (x, i) and box (x, j) are adjacent (|i - j| = 1)
    bool horz_adj(int x, int i, int j)
    {
        if (i > j)
            i = j;
        return (((box[x][i] >> 3) & 1) == 0);
    }

    // check if box (i, y) and box (j, y) are adjacent (|i - j| = 1)
    bool vert_adj(int y, int i, int j)
    {
        if (i > j)
            i = j;
        return (((box[i][y] >> 1) & 1) == 0);
    }

    // count tgenes number of free edges in box (i, j)
    int free_edges(int i, int j)
    {
        return 4 - __builtin_popcount(box[i][j]);
    }

    // check if box (x, y) is visited, or not exist
    bool banned_vis(int x, int y)
    {
        return (x < 1 || y < 1 || x > sq_size || y > sq_size || vis[x][y]);
    }

    bool is_chain_start(int x, int y)
    {
        if (free_edges(x, y) != 2)
            return 0;
        if (vis[x][y])
            return 0;
        int cnt = 0;
        for (int l = 0; l <= 3; l++)
        {
            int u = x + dx[l], v = y + dy[l];
            if (banned_vis(u, v))
                continue;
            // check if adjacent
            if (dx[l] == 0 && !horz_adj(x, y, v))
                continue;
            if (dy[l] == 0 && !vert_adj(y, x, u))
                continue;

            // adjacent boxes in a chain must be adjacent and both have 2 free edges
            if (free_edges(u, v) == 2)
                cnt++;
        }
        return (cnt <= 1);
    }

    int dfs(int x, int y)
    {
        vis[x][y] = 1;
        // visit 4 sides
        for (int l = 0; l <= 3; l++)
        {
            int u = x + dx[l], v = y + dy[l];
            // check if adjacent
            if (dx[l] == 0 && !horz_adj(x, y, v))
                continue;
            if (dy[l] == 0 && !vert_adj(y, x, u))
                continue;

            // adjacent boxes in a chain must be adjacent and both have 2 free edges
            if (!banned_vis(u, v) && free_edges(u, v) == 2)
                return dfs(u, v) + 1;
        }
        return 1;
    }

    void variable_cal()
    {
        long_chain = short_chain = double_cross = 0;
        for (int i = 1; i <= sq_size; i++)
            for (int j = 1; j <= sq_size; j++)
                vis[i][j] = 0;
        for (int i = 1; i <= sq_size; i++)
            for (int j = 1; j <= sq_size; j++)
                if (is_chain_start(i, j))
                {
                    if (dfs(i, j) <= 2)
                        short_chain++;
                    else
                        long_chain++;
                }
        for (int i = 1; i < sq_size; i++)
            for (int j = 1; j <= sq_size; j++)
                if (box[i][j] == 13 && box[i + 1][j] == 14)
                    double_cross++;

        for (int i = 1; i <= sq_size; i++)
            for (int j = 1; j < sq_size; j++)
                if (box[i][j] == 7 && box[i][j + 1] == 11)
                    double_cross++;

        box3 = box4 = 0;
        for (int i = 1; i <= sq_size; i++)
            for (int j = 1; j <= sq_size; j++)
                if (free_edges(i, j) == 3)
                    box3++;
                else if (free_edges(i, j) == 4)
                    box4++;
    }

    long long v_eval(long long A)
    {
        variable_cal();

        long long v_res = 0;

        long long S1 = (A & 63);
        A >>= 6;
        long long S2 = (A & 63);
        A >>= 6;

        if (long_chain % 2)
            v_res += S1;
        else
            v_res += S2;

        S1 = (A & 63);
        A >>= 6;
        S2 = (A & 63);
        A >>= 6;

        if (short_chain % 2)
            v_res += S1;
        else
            v_res += S2;

        S1 = (A & 63);
        A >>= 6;
        S2 = (A & 63);
        A >>= 6;

        if (double_cross % 2)
            v_res += S1;
        else
            v_res += S2;

        S1 = (A & 63);
        A >>= 6;
        S2 = (A & 63);
        A >>= 6;

        if (box3 % 2)
            v_res += S1;
        else
            v_res += S2;

        S1 = (A & 63);
        A >>= 6;
        S2 = (A & 63);
        A >>= 6;

        if (box4 % 2)
            v_res += S1;
        else
            v_res += S2;

        return v_res;
    }

    // Game initialization
    bool human_move(int id)
    {
        if (valid_move[id] == 1 || id >= 2 * sq_size * (sq_size + 1))
        {
            display();
            cout << "invalid move. move again!" << "\n";
            return 0;
        }
        move(id);
        int cnt = 0;
        for (int i = 1; i <= sq_size; i++)
            for (int j = 1; j <= sq_size; j++)
                if (check_if_filled(i, j))
                {
                    cnt++;
                    if (dis[i][j] == ' ')
                        dis[i][j] = 'o';
                }
        if (cnt > p1_score + p2_score)
            p1_score = cnt - p2_score;
        else
            move_count++;
        return 1;
    }

    int count_takeable_squares()
    {
        int res = 0;
        for (int i = 1; i <= sq_size; i++)
            for (int j = 1; j <= sq_size; j++)
                if (free_edges(i, j) == 1)
                    res++;
        return res;
    }

    int count_done_squares()
    {
        int res = 0;
        for (int i = 1; i <= sq_size; i++)
            for (int j = 1; j <= sq_size; j++)
                if (check_if_filled(i, j) == 1)
                    res++;
        return res;
    }

    bool random_1_move()
    {
        vector<int> vt;
        for (int i = 0; i < 2 * sq_size * (sq_size + 1); i++)
            if (valid_move[i] == 0)
                vt.push_back(i);
        int choice = rng(0, vt.size() - 1);
        move(vt[choice]);
        int cnt = 0;
        for (int i = 1; i <= sq_size; i++)
            for (int j = 1; j <= sq_size; j++)
                if (check_if_filled(i, j))
                {
                    cnt++;
                    if (dis[i][j] == ' ')
                        dis[i][j] = 'o';
                }
        if (cnt > p1_score + p2_score)
            p1_score = cnt - p2_score;
        else
            move_count++;
        return 1;
    }

    bool random_2_move()
    {
        vector<int> vt;
        for (int i = 0; i < 2 * sq_size * (sq_size + 1); i++)
            if (valid_move[i] == 0)
                vt.push_back(i);
        int choice = rng(0, vt.size() - 1);
        move(vt[choice]);
        int cnt = 0;
        for (int i = 1; i <= sq_size; i++)
            for (int j = 1; j <= sq_size; j++)
                if (check_if_filled(i, j))
                {
                    cnt++;
                    if (dis[i][j] == ' ')
                        dis[i][j] = 'o';
                }
        if (cnt > p1_score + p2_score)
            p2_score = cnt - p1_score;
        else
            move_count++;
        return 1;
    }

    bool AI_1_move(long long A)
    {
        int choice = -1, maxV = -1, ok = 0;
        int ts = count_takeable_squares();
        int ds = count_done_squares();
        for (int i = 0; i < 2 * sq_size * (sq_size + 1); i++)
            if (valid_move[i] == 0)
            {
                move(i);
                long long cur = v_eval(A);
                if (ds < count_done_squares())
                    cur += 63;
                bool ok2 = (ts >= count_takeable_squares());
                if (ok2 > ok)
                {
                    ok = ok2;
                    maxV = cur;
                    choice = i;
                }
                else if (cur > maxV)
                {
                    maxV = cur;
                    choice = i;
                }
                unmove(i);
            }
        move(choice);
        int cnt = 0;
        for (int i = 1; i <= sq_size; i++)
            for (int j = 1; j <= sq_size; j++)
                if (check_if_filled(i, j))
                {
                    cnt++;
                    if (dis[i][j] == ' ')
                        dis[i][j] = 'x';
                }
        if (cnt > p1_score + p2_score)
            p1_score = cnt - p2_score;
        else
            move_count++;
        return 1;
    }

    bool AI_2_move(long long B)
    {
        int choice = -1, maxV = -1, ok = 0;
        int ts = count_takeable_squares();
        int ds = count_done_squares();
        for (int i = 0; i < 2 * sq_size * (sq_size + 1); i++)
            if (valid_move[i] == 0)
            {
                move(i);
                long long cur = v_eval(B);
                if (ds < count_done_squares())
                    cur += 63;
                bool ok2 = (ts >= count_takeable_squares());
                if (ok2 > ok)
                {
                    ok = ok2;
                    maxV = cur;
                    choice = i;
                }
                else if (cur > maxV)
                {
                    maxV = cur;
                    choice = i;
                }
                unmove(i);
            }
        move(choice);
        int cnt = 0;
        for (int i = 1; i <= sq_size; i++)
            for (int j = 1; j <= sq_size; j++)
                if (check_if_filled(i, j))
                {
                    cnt++;
                    if (dis[i][j] == ' ')
                        dis[i][j] = 'x';
                }
        if (cnt > p1_score + p2_score)
            p2_score = cnt - p1_score;
        else
            move_count++;
        return 1;
    }

    bool p2_bot_greedy_move()
    {
        bool gr = 0;
        for (int i = 1; i <= sq_size; i++)
            if (gr == 0)
                for (int j = 1; j <= sq_size; j++)
                    if (free_edges(i, j) == 1)
                    {
                        if ((box[i][j] & 1) == 0)
                        {
                            int id = (i - 1) * sq_size + j - 1;
                            // cout << "!U " << i << " " << j << " " << id << "\n";
                            move(id);
                            gr = 1;
                            break;
                        }
                        else if (((box[i][j] >> 1) & 1) == 0)
                        {
                            int id;
                            if (i < sq_size)
                                id = i * sq_size + j - 1;
                            else
                                id = 2 * sq_size * sq_size + j - 1;
                            // cout << "!D " << i << " " << j << " " << id << "\n";
                            move(id);
                            gr = 1;
                            break;
                        }
                        else if (((box[i][j] >> 2) & 1) == 0)
                        {
                            int id = sq_size * sq_size + (i - 1) * sq_size + j - 1;
                            // cout << "!L " << i << " " << j << " " << id << "\n";
                            move(id);
                            gr = 1;
                            break;
                        }
                        else
                        {
                            int id;
                            if (j < sq_size)
                                id = sq_size * sq_size + (i - 1) * sq_size + j;
                            else
                                id = 2 * sq_size * sq_size + sq_size + i - 1;
                            // cout << "!R " << i << " " << j << " " << id << "\n";
                            move(id);
                            gr = 1;
                            break;
                        }
                    }
        if (!gr)
        {
            // wrong move: nếu box có 2 cạnh thì không nên đặt cạnh thứ 3 vào, trừ khi bắt buộc
            vector<int> wrong_move;
            wrong_move.resize(2 * (sq_size) * (sq_size + 1));
            fill(wrong_move.begin(), wrong_move.end(), 0);
            for (int i = 1; i <= sq_size; i++)
                for (int j = 1; j <= sq_size; j++)
                    if (free_edges(i, j) == 2)
                    {
                        if ((box[i][j] & 1) == 0)
                        {
                            int id = (i - 1) * sq_size + j - 1;
                            wrong_move[id] = 1;
                        }
                        if (((box[i][j] >> 1) & 1) == 0)
                        {
                            int id;
                            if (i < sq_size)
                                id = i * sq_size + j - 1;
                            else
                                id = 2 * sq_size * sq_size + j - 1;
                            wrong_move[id] = 1;
                        }
                        if (((box[i][j] >> 2) & 1) == 0)
                        {
                            int id = sq_size * sq_size + (i - 1) * sq_size + j - 1;
                            wrong_move[id] = 1;
                        }
                        if (((box[i][j] >> 3) & 1) == 0)
                        {
                            int id;
                            if (j < sq_size)
                                id = sq_size * sq_size + (i - 1) * sq_size + j;
                            else
                                id = 2 * sq_size * sq_size + sq_size + i - 1;
                            wrong_move[id] = 1;
                        }
                    }
            vector<int> cur;
            int wrong_move_count = 0;
            for (int i = 0; i <= 2 * sq_size * (sq_size + 1) - 1; i++)
                if (!valid_move[i])
                    wrong_move_count += 1 - wrong_move[i];
            if (wrong_move_count == 0)
            {
                for (int i = 0; i <= 2 * sq_size * (sq_size + 1) - 1; i++)
                    if (!valid_move[i])
                        cur.push_back(i);
            }
            else
            {
                for (int i = 0; i <= 2 * sq_size * (sq_size + 1) - 1; i++)
                    if (!valid_move[i] && !wrong_move[i])
                        cur.push_back(i);
            }
            int choice = rng(0, cur.size() - 1);
            move(cur[choice]);
        }
        int cnt = 0;
        for (int i = 1; i <= sq_size; i++)
            for (int j = 1; j <= sq_size; j++)
                if (check_if_filled(i, j))
                {
                    cnt++;
                    if (dis[i][j] == ' ')
                        dis[i][j] = 'x';
                }
        if (cnt > p1_score + p2_score)
            p2_score = cnt - p1_score;
        else
            move_count++;
        return 1;
    }

    bool p1_bot_greedy_move()
    {
        bool gr = 0;
        for (int i = 1; i <= sq_size; i++)
            if (gr == 0)
                for (int j = 1; j <= sq_size; j++)
                    if (free_edges(i, j) == 1)
                    {
                        if ((box[i][j] & 1) == 0)
                        {
                            int id = (i - 1) * sq_size + j - 1;
                            // cout << "!U " << i << " " << j << " " << id << "\n";
                            move(id);
                            gr = 1;
                            break;
                        }
                        else if (((box[i][j] >> 1) & 1) == 0)
                        {
                            int id;
                            if (i < sq_size)
                                id = i * sq_size + j - 1;
                            else
                                id = 2 * sq_size * sq_size + j - 1;
                            // cout << "!D " << i << " " << j << " " << id << "\n";
                            move(id);
                            gr = 1;
                            break;
                        }
                        else if (((box[i][j] >> 2) & 1) == 0)
                        {
                            int id = sq_size * sq_size + (i - 1) * sq_size + j - 1;
                            // cout << "!L " << i << " " << j << " " << id << "\n";
                            move(id);
                            gr = 1;
                            break;
                        }
                        else
                        {
                            int id;
                            if (j < sq_size)
                                id = sq_size * sq_size + (i - 1) * sq_size + j;
                            else
                                id = 2 * sq_size * sq_size + sq_size + i - 1;
                            // cout << "!R " << i << " " << j << " " << id << "\n";
                            move(id);
                            gr = 1;
                            break;
                        }
                    }
        if (!gr)
        {
            vector<int> wrong_move;
            wrong_move.resize(2 * (sq_size) * (sq_size + 1));
            fill(wrong_move.begin(), wrong_move.end(), 0);
            for (int i = 1; i <= sq_size; i++)
                for (int j = 1; j <= sq_size; j++)
                    if (free_edges(i, j) == 2)
                    {
                        if ((box[i][j] & 1) == 0)
                        {
                            int id = (i - 1) * sq_size + j - 1;
                            wrong_move[id] = 1;
                        }
                        if (((box[i][j] >> 1) & 1) == 0)
                        {
                            int id;
                            if (i < sq_size)
                                id = i * sq_size + j - 1;
                            else
                                id = 2 * sq_size * sq_size + j - 1;
                            wrong_move[id] = 1;
                        }
                        if (((box[i][j] >> 2) & 1) == 0)
                        {
                            int id = sq_size * sq_size + (i - 1) * sq_size + j - 1;
                            wrong_move[id] = 1;
                        }
                        if (((box[i][j] >> 3) & 1) == 0)
                        {
                            int id;
                            if (j < sq_size)
                                id = sq_size * sq_size + (i - 1) * sq_size + j;
                            else
                                id = 2 * sq_size * sq_size + sq_size + i - 1;
                            wrong_move[id] = 1;
                        }
                    }
            vector<int> cur;
            int wrong_move_count = 0;
            for (int i = 0; i <= 2 * sq_size * (sq_size + 1) - 1; i++)
                if (!valid_move[i])
                    wrong_move_count += 1 - wrong_move[i];
            if (wrong_move_count == 0)
            {
                for (int i = 0; i <= 2 * sq_size * (sq_size + 1) - 1; i++)
                    if (!valid_move[i])
                        cur.push_back(i);
            }
            else
            {
                for (int i = 0; i <= 2 * sq_size * (sq_size + 1) - 1; i++)
                    if (!valid_move[i] && !wrong_move[i])
                        cur.push_back(i);
            }
            int choice = rng(0, cur.size() - 1);
            move(cur[choice]);
        }
        int cnt = 0;
        for (int i = 1; i <= sq_size; i++)
            for (int j = 1; j <= sq_size; j++)
                if (check_if_filled(i, j))
                {
                    cnt++;
                    if (dis[i][j] == ' ')
                        dis[i][j] = 'o';
                }
        if (cnt > p1_score + p2_score)
            p1_score = cnt - p2_score;
        else
            move_count++;
        return 1;
    }

    bool is_finished()
    {
        if (p1_score + p2_score == sq_size * sq_size)
        {
            return 1;
        }
        return 0;
    }

    void display()
    {
        cout << "\n\n\n\n\n";
        for (int i = 1; i <= sq_size; i++)
        {
            for (int j = 1; j <= sq_size; j++)
            {
                cout << "." << " ";
                if (box[i][j] & 1)
                    cout << "___" << " ";
                else
                    cout << "   " << " ";
            }
            cout << "." << "\n\n";
            for (int j = 1; j <= sq_size; j++)
            {
                if (box[i][j] & 4)
                    cout << "|" << " ";
                else
                    cout << " " << " ";
                cout << " " << dis[i][j] << "  ";
            }
            if (box[i][sq_size] & 8)
                cout << "|" << "\n\n";
            else
                cout << " " << "\n\n";
        }

        for (int i = 1; i <= sq_size; i++)
        {
            cout << "." << " ";
            if (box[sq_size][i] & 2)
                cout << "___" << " ";
            else
                cout << "   " << " ";
        }
        cout << "." << "\n\n\n";
        cout << "____________\n\n";
        cout << "o score: " << p1_score << "\n";
        cout << "x score: " << p2_score << "\n";
        cout << "____________\n";
    }
};
GeneticInterface cur;

void printvalues(long long A)
{
    for (int i = 0; i < 10; i++)
    {
        int c = (A & 63);
        cout << format(c) << " ";
        A >>= 6;
    }
}
int display_AI_random_battle(GeneticInterface cur, long long A1)
{
    cur.set_size(3);
    cur.display();
    bool cur_move = 0;
    while (true)
    {
        if (cur_move == 0)
        {
            int c = cur.p1_score;
            cur.AI_1_move(A1);
            // waiting process
            int h = 1;
            for (int i = 0; i <= 2000000000; i++)
                h = 1;
            if (cur.p1_score == c)
                cur_move = 1;
            cout << "AI moved\n";
        }
        else
        {
            int c = cur.p2_score;
            cur.random_2_move();
            // waiting process
            int h = 1;
            for (int i = 0; i <= 2000000000; i++)
                h = 1;
            if (cur.p2_score == c)
                cur_move = 0;
            cout << "Random moved\n";
        }
        cur.display();
        if (cur.is_finished())
            break;
    }
    if (cur.p1_score > cur.p2_score)
        cout << "AI win!";
    else if (cur.p1_score < cur.p2_score)
        cout << "Random win!";
    else
        cout << "draw!";
    if (cur.p1_score > cur.p2_score)
        return 1;
    else if (cur.p1_score < cur.p2_score)
        return 2;
    else
        return 0;
}

int display_AI_greedy_battle(GeneticInterface cur, long long A1)
{
    cur.set_size(3);
    cur.display();
    bool cur_move = 0;
    while (true)
    {
        if (cur_move == 0)
        {
            int c = cur.p1_score;
            cur.AI_1_move(A1);
            // waiting process
            int h = 1;
            for (int i = 0; i <= 2000000000; i++)
                h = 1;
            if (cur.p1_score == c)
                cur_move = 1;
            cout << "AI moved\n";
        }
        else
        {
            int c = cur.p2_score;
            cur.p2_bot_greedy_move();
            // waiting process
            int h = 1;
            for (int i = 0; i <= 2000000000; i++)
                h = 1;
            if (cur.p2_score == c)
                cur_move = 0;
            cout << "Greedy moved\n";
        }
        cur.display();
        if (cur.is_finished())
            break;
    }
    if (cur.p1_score > cur.p2_score)
        cout << "AI win!";
    else if (cur.p1_score < cur.p2_score)
        cout << "Greedy win!";
    else
        cout << "draw!";
    if (cur.p1_score > cur.p2_score)
        return 1;
    else if (cur.p1_score < cur.p2_score)
        return 2;
    else
        return 0;
}

int human_battle(GeneticInterface cur, long long A1)
{
    cur.set_size(3);
    cur.display();
    bool cur_move = 0;
    while (true)
    {
        if (cur_move == 0)
        {
            int c = cur.p1_score;
            cout << "Enter your move. Move is a number from 0 to 23\n";
            int k;
            cin >> k;
            while (cur.human_move(k) == 0)
            {
                cin >> k;
            }
            if (cur.p1_score == c)
                cur_move = 1;
        }
        else
        {
            int c = cur.p2_score;
            cur.AI_2_move(A1);
            // waiting process
            int h = 1;
            for (int i = 0; i <= 2000000000; i++)
                h = 1;
            if (cur.p2_score == c)
                cur_move = 0;
        }
        cur.display();
        if (cur.is_finished())
            break;
    }
    if (cur.p1_score > cur.p2_score)
        cout << "you win!";
    else if (cur.p1_score < cur.p2_score)
        cout << "AI win!";
    else
        cout << "draw!";
    if (cur.p1_score > cur.p2_score)
        return 1;
    else if (cur.p1_score < cur.p2_score)
        return 2;
    else
        return 0;
}

int battle(GeneticInterface cur, long long A1, long long A2)
{
    cur.set_size(3);
    bool cur_move = 0;
    while (true)
    {
        if (cur_move == 0)
        {
            int c = cur.p1_score;
            cur.AI_1_move(A1);
            if (cur.p1_score == c)
                cur_move = 1;
        }
        else
        {
            int c = cur.p2_score;
            cur.AI_2_move(A2);
            if (cur.p2_score == c)
                cur_move = 0;
        }
        // cur.display();
        if (cur.is_finished())
            break;
    }
    if (cur.p1_score > cur.p2_score)
        return 1;
    else if (cur.p1_score < cur.p2_score)
        return 2;
    else
        return 0;
}

map<long long, long long> mpw, mpd, mprw, mprd;
int random_battle(GeneticInterface cur, long long A1, bool ok)
{
    int W = 0, D = 0, L = 0;
    for (int i = 1; i <= 50; i++)
    {
        cur.set_size(3);
        bool cur_move = 0;
        while (true)
        {
            if (cur_move == 0)
            {
                int c = cur.p1_score;
                cur.AI_1_move(A1);
                if (cur.p1_score == c)
                    cur_move = 1;
            }
            else
            {
                int c = cur.p2_score;
                cur.random_2_move();
                if (cur.p2_score == c)
                    cur_move = 0;
            }
            // cur.display();
            if (cur.is_finished())
                break;
        }
        if (cur.p1_score > cur.p2_score)
            W++;
        else if (cur.p2_score > cur.p1_score)
            L++;
        else
            D++;
        cur.set_size(3);
        cur_move = 0;
        while (true)
        {
            if (cur_move == 0)
            {
                int c = cur.p1_score;
                cur.random_1_move();
                if (cur.p1_score == c)
                    cur_move = 1;
            }
            else
            {
                int c = cur.p2_score;
                cur.AI_2_move(A1);
                if (cur.p2_score == c)
                    cur_move = 0;
            }
            // cur.display();
            if (cur.is_finished())
                break;
        }
        if (cur.p1_score > cur.p2_score)
            L++;
        else if (cur.p2_score > cur.p1_score)
            W++;
        else
            D++;
    }
    if (ok)
    {
        cout << "Random bot record (W / D / L): " << W << " / " << D << " / " << L << "\n";
    }
    mprw[A1] = W;
    mprd[A1] = D;
    return W * 2 + D;
}

int greedy_battle(GeneticInterface cur, long long A1, bool ok)
{
    int W = 0, D = 0, L = 0;
    for (int i = 1; i <= 500; i++)
    {
        cur.set_size(3);
        bool cur_move = 0;
        while (true)
        {
            if (cur_move == 0)
            {
                int c = cur.p1_score;
                cur.AI_1_move(A1);
                if (cur.p1_score == c)
                    cur_move = 1;
            }
            else
            {
                int c = cur.p2_score;
                cur.p2_bot_greedy_move();
                if (cur.p2_score == c)
                    cur_move = 0;
            }
            // cur.display();
            if (cur.is_finished())
                break;
        }
        if (cur.p1_score > cur.p2_score)
            W++;
        else if (cur.p2_score > cur.p1_score)
            L++;
        else
            D++;
        cur.set_size(3);
        cur_move = 0;
        while (true)
        {
            if (cur_move == 0)
            {
                int c = cur.p1_score;
                cur.p1_bot_greedy_move();
                if (cur.p1_score == c)
                    cur_move = 1;
            }
            else
            {
                int c = cur.p2_score;
                cur.AI_2_move(A1);
                if (cur.p2_score == c)
                    cur_move = 0;
            }
            // cur.display();
            if (cur.is_finished())
                break;
        }
        if (cur.p1_score > cur.p2_score)
            L++;
        else if (cur.p2_score > cur.p1_score)
            W++;
        else
            D++;
    }
    if (ok)
    {
        cout << "Greedy bot record (W / D / L): " << W << " / " << D << " / " << L << "\n";
    }
    mpw[A1] = W;
    mpd[A1] = D;
    return W * 2 + D;
}
struct prof
{
    int w, d, l, b;
    long long seed;

    prof()
    {
        w = d = l = b = 0;
    }

    void prep()
    {
        seed = rng(0, (1LL << 60) - 1);
        w = d = l = b = 0;
    }

    int points()
    {
        return w * 6 + d * 2 - l * 4 + b;
    }

    void print()
    {
        printvalues(seed);
        cout << " | " << "Wins: " << format(w) << " " << "Draws: " << format(d) << " " << "Loss: " << format(l) << " | Points: " << w * 6 + d * 2 - l * 4 + b << "\n";
        random_battle(cur, seed, 0);
        cout << "Greedy bot record (W / D / L): " << mpw[seed] << " / " << mpd[seed] << " / " << 1000 - mpw[seed] - mpd[seed] << "\n";
        cout << "Random bot record (W / D / L): " << mprw[seed] << " / " << mprd[seed] << " / " << 100 - mprw[seed] - mprd[seed] << "\n";
    }
};

long long crossover(long long a, long long b)
{
    long long res = 0, tmp = 1;
    for (int i = 0; i < 10; i++)
    {
        long long A = (a & 63);
        long long B = (b & 63);
        int rate = rng(0, 1);
        if (rate == 0)
            res += A * tmp;
        else
            res += B * tmp;
        tmp *= (1LL << 6);
        a >>= 6;
        b >>= 6;
    }
    return res;
}

long long mutate(long long a)
{
    long long res = 0, tmp = 1;
    int rate = rng(0, 9);
    for (int i = 0; i < 10; i++)
    {
        long long A = (a & 63);
        if (rate == i)
            A = rng(0, 63);
        res += A * tmp;
        tmp *= (1LL << 6);
        a >>= 6;
    }
    return res;
}

bool cmp(prof a, prof b)
{
    if (a.points() != b.points())
        return (a.points() > b.points());
    if (a.w != b.w)
        return (a.w > b.w);
    return (a.d > b.d);
}

const int N = 40;
prof genes[N + 2];
long long utilize(bool ok)
{
    int rate = rng(0, 99);
    /*
    45% crossover
    45% mutate
    10% new generation
    */
    if (rate < 45)
    {
        int x = rng(0, N / 2 - 2);
        int y = rng(x + 1, N / 2 - 1);
        prof z;
        z.seed = crossover(genes[x].seed, genes[y].seed);
        if (ok)
        {
            cout << "crossed " << x + 1 << " and " << y + 1 << " gene:\n";
            z.print();
        }
        return z.seed;
    }
    else if (rate < 90)
    {

        int x = rng(0, N / 2 - 1);
        prof z;
        z.seed = mutate(genes[x].seed);
        if (ok)
        {
            cout << "mutated " << x + 1 << " gene:\n";
            z.print();
        }
        return z.seed;
    }
    else
    {
        prof c;
        c.seed = rng(0, (1LL << 60) - 1);
        if (ok)
        {
            cout << "new seed: \n";
            c.print();
        }
        return c.seed;
    }
}
int main()
{
    int generation_num = 5; // Số thế hệ
    cout << "Genetic Algorithm Visualization.\n";
    cout << N << " seeds will be participating in a double round-robin.\nAfter each generation, " << N / 2 << " seeds with the least fitness score (points) will be replaced, either by mutation, crossover or new seed randomly.\n";
    cout << "There will be " << generation_num << " generations in total.\n";
    cout << "Type any string then press enter to begin: \n";
    string s;
    cin >> s;
    cout << "Currently generating...\n";
    for (int i = 0; i < N; i++)
        genes[i].prep();
    for (int time = 1; time <= generation_num; time++)
    {
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                if (i != j)
                {
                    int res = battle(cur, genes[i].seed, genes[j].seed);
                    if (res == 1)
                        genes[i].w++, genes[j].l++;
                    else if (res == 2)
                        genes[i].l++, genes[j].w++;
                    else
                        genes[i].d++, genes[j].d++;
                }

        for (int i = 0; i < N; i++)
        {
            genes[i].b = greedy_battle(cur, genes[i].seed, 0);
        }

        sort(genes, genes + N, cmp);

        cout << "gen count: " << time << "\n";
        if (time < generation_num)
        {
            for (int i = N - N / 2; i < N; i++)
                genes[i].seed = utilize(0);
            for (int i = 0; i < N; i++)
                genes[i].w = genes[i].d = genes[i].l = 0;
        }
    }
    cout << N / 2 << " Best seeds after " << generation_num << " generations: \n";
    for (int i = 0; i < N / 2; i++)
    {
        cout << format(i + 1) << " | ";
        genes[i].print();
    }
    cout << "________________________________________";
    cout << "\n\n";

    cout << "Best seed: \n";
    cout << "Decimal value: " << genes[0].seed << "\n";
    cout << "Elemental parts and basic informations: ";
    genes[0].print();
    cout << "\n";
    cout << "Type 0 for Greedy vs AI, 1 for Human vs AI, 2 for Random vs AI: \n";
    cin >> s;
    while (s != "0" && s != "1" && s != "2")
    {
        cout << "Invalid request\n";
        cin >> s;
    }
    if (s == "1")
    {
        cout << "Human battle with the best seed. o for human, x for AI. o goes first. Type any string then press enter to begin: \n";
        cin >> s;
        human_battle(cur, genes[0].seed);
    }
    else if (s == "0")
    {
        cout << "Greedy bot with the best seed. o for AI, x for bot. o goes first. Type any string then press enter to begin: \n";
        cin >> s;
        display_AI_greedy_battle(cur, genes[0].seed);
    }
    else if (s == "2")
    {
        cout << "Random bot with the best seed. o for AI, x for bot. o goes first. Type any string then press enter to begin: \n";
        cin >> s;
        display_AI_random_battle(cur, genes[0].seed);
    }
}
/*
Suggest cải tiến AI: kết hợp seed với greedy / minimax để tối ưu kết quả
*/