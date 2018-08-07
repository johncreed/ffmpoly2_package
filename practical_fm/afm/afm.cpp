#include "afm.h"

extern AfmIdx nr_threads;
extern AfmIdx subsample_rate;
extern AfmIdx max_nt_iters;
extern AfmVal nt_eps;
extern bool is_relative_cg;

Parameter get_default_parameter()
{
    Parameter param;
    param.solver = 1;
    param.lambda_w = 1;
    param.lambda_U = 4;
    param.lambda_V = 4;
    param.K = 4;
    param.iters = 20;
    param.tol = 0;
    param.nu = 0.01;
    param.kappa = 0.5;
    param.std = 0.1;
    param.cg_iters = 10;
    param.eta = 0.1;
    return param;
}

void read_csr_problem(string filename, AfmData &data)
{
    AfmIdx nr_instances = 0;
    AfmIdx nr_features = 0;
    AfmIdx nr_nodes = 0;
    AfmIdx max_nnz = 0;
    ifstream fs(filename);

    string line;
    vector<AfmIdx> nnzs;
    while (getline(fs, line))
    {
        AfmVal label = 0;
        istringstream iss(line);
        iss >> label;
        if (label > 0)
            label = 1;
        else
            label = -1;
        data.labels.push_back(label);
        nr_instances++;
        AfmIdx j = 0;
        AfmVal x = 0;
        char c;
        AfmIdx nnz = 0;
        while(iss >> j >> c >> x)
        {
            if (j > nr_features)
                nr_features = j;
            nr_nodes++;
            nnz++;
            Node node;
            node.idx = j;
            node.val = x;
            data.nodes.push_back(node);
        }
        nnzs.push_back(nnz);
        if (nnz > max_nnz)
            max_nnz = nnz;
    }

    data.I = nr_instances;
    data.J = nr_features+1;
    data.labels.resize(nr_instances);
    data.indexes.resize(nr_instances+1);

    cout << "# of ins.: " << nr_instances << ", # of feat.: " << nr_features << endl;
    cout << "average/max/total nnz's per ins.: "
         << nr_nodes/nr_instances << "/"
         << max_nnz << "/" << nr_nodes << endl;
    fs.close();


    Node *current = data.nodes.data();
    data.indexes[0] = current;
    for (AfmIdx i = 0; i < data.I; i++)
    {
        for (AfmIdx nz = 0; nz < nnzs[i]; nz++)
            current++;
        data.indexes[i+1] = current;
    }
}

void read_csc_problem(string filename, AfmData &data)
{
    AfmIdx nr_instances = 0;
    AfmIdx max_features = 0;
    AfmIdx nr_nodes = 0;
    AfmIdx max_nnz = 0;
    ifstream fs(filename);

    string line;
    while (getline(fs, line))
    {
        AfmVal label = 0;
        istringstream iss(line);
        iss >> label;
        nr_instances++;
        AfmIdx j = 0;
        AfmVal x = 0;
        char c;
        AfmIdx nnz = 0;
        while(iss >> j >> c >> x)
        {
            if (j > max_features)
                max_features = j;
            nr_nodes++;
            nnz++;
        }
        if (nnz > max_nnz)
            max_nnz = nnz;
    }

    data.I = nr_instances;
    data.J = max_features+1;
    data.labels.resize(nr_instances);
    data.indexes.resize(max_features+2);
    data.nodes.resize(nr_nodes);

    cout << "# of ins.: " << nr_instances << ", max of feat.: " << max_features << endl;
    cout << "average/max/total nnz's per ins.: "
         << nr_nodes/nr_instances << "/"
         << max_nnz << "/" << nr_nodes << endl;

    const AfmIdx J = data.J;
    fs.clear();
    fs.seekg(0, ios::beg);

    vector<AfmIdx> freqs(max_features+2, 0);
    while (getline(fs, line))
    {
        istringstream iss(line);
        AfmVal label = 0;
        AfmIdx j = 0;
        AfmVal x = 0;
        char c;
        iss >> label;
        while(iss >> j >> c >> x)
            freqs[j+1]++;
    }

    for (AfmIdx j = 1; j < J+1; j++)
        freqs[j] = freqs[j]+freqs[j-1];

    for (AfmIdx j = 0; j < J+1; j++)
        data.indexes[j] = data.nodes.data()+freqs[j];

    vector<Node*> col_indexes = data.indexes;

    fs.clear();
    fs.seekg(0, ios::beg);

    for (AfmIdx i = 0; getline(fs, line); i++)
    {
        AfmVal label = 0;
        istringstream iss(line);
        iss >> label;
        if (label > 0)
            data.labels[i] = 1;
        else
            data.labels[i] = -1;
        AfmIdx j = 0;
        AfmVal x = 0;
        char c;
        while(iss >> j >> c >> x)
        {
            col_indexes[j]->idx = i;
            col_indexes[j]->val = x;
            col_indexes[j]++;
        }
    }
    fs.close();
}

void initialize_model(const AfmVal std, const AfmIdx solver,
    const AfmIdx K, const AfmIdx J, AfmModel &model)
{
    model.solver = solver;
    model.K = K;
    model.J = J;
    model.w = vector<AfmVal>(J, 0);
    model.U = vector<AfmVal>(K*J, 0);
    default_random_engine engine(0);
    //uniform_real_distribution<AfmVal> distribution(-std/sqrt(K), std/sqrt(K));
    uniform_real_distribution<AfmVal> distribution(0, 1.0/sqrt(K));

    if (solver == AFM_ADAGRAD ||
        solver == AFM_ALS_PRIMAL)
    {
        model.V = vector<AfmVal>(K*J, 0);
        for (AfmIdx j = 0; j < J; j++)
            for (AfmIdx k = 0; k < K; k++)
            {
                model.U[j*K+k] = distribution(engine);
                model.V[j*K+k] = distribution(engine);
            }
    }
    else if (solver == FM_AGD ||
             solver == FM_GD ||
             solver == FM_ADAGRAD)
    {
        for (AfmIdx j = 0; j < J; j++)
            for (AfmIdx k = 0; k < K; k++)
                model.U[j*K+k] = distribution(engine);
    }
}

AfmVal calc_xi_no_self(const AfmVal &q_k,
    const AfmVal &v_jk, const AfmVal &x_ij)
{ return (q_k-v_jk*x_ij)*x_ij; }

AfmVal calc_xi_self(const AfmVal &q_k,
    const AfmVal &v_jk, const AfmVal &x_ij)
{ return q_k*x_ij; }

void pcg_w(const AfmVal lambda,
    const AfmData &data,
    const vector<AfmVal> &sigma,
    const vector<AfmVal> &freqs,
    const bool do_pcond,
    const bool do_gd,
    const AfmIdx cg_iters,
    const vector<AfmVal> &g_w,
    const vector<AfmIdx> orders,
    vector<AfmVal> &s_w)
{
    if (do_gd)
    {
        for (AfmIdx j = 0; j < g_w.size(); j++)
            s_w[j] = -g_w[j];
        return;
    }
    else
    {
        for (AfmIdx j = 0; j < g_w.size(); j++)
            s_w[j] = 0;
    }

    AfmIdx J = g_w.size();
    vector<AfmVal> M(J, 0);
    vector<AfmVal> H_w_d(J, 0);
    vector<AfmVal> reduction_array(nr_threads*J, 0);
    vector<AfmVal> g_hat(J, 0);
    vector<AfmVal> d(J, 0);
    vector<AfmVal> r(J, 0);
    AfmVal alpha = 0;
    AfmVal beta = 0;
    AfmVal gamma = 0;
    AfmVal r_square = 0.0;
    AfmVal g_w_square = 0.0;
    vector<Node*> X = data.indexes;

    uniform_int_distribution<AfmIdx> dist(0, data.I-1);
    minstd_rand0 generator(rand());

    if (do_pcond)
    {
#pragma omp parallel for num_threads(nr_threads) schedule(guided)
        for (AfmIdx i_ = 0; i_ < data.I/subsample_rate; i_++)
        {
            AfmIdx tid = omp_get_thread_num();
            AfmIdx i = orders[i_];

            for (Node *x = X[i]; x != X[i+1]; ++x)
            {
                const AfmIdx j = x->idx;
                const AfmVal x_ij = x->val;
                reduction_array[tid*J+j] += sigma[i]*(1-sigma[i])*x_ij*x_ij;
            }
        }
#pragma omp parallel for num_threads(nr_threads) schedule(static) reduction(+:r_square, g_w_square)
        for (AfmIdx j = 0; j < J; j++)
        {
            for(AfmIdx tid = 0; tid < nr_threads; tid++)
            {
                M[j] += reduction_array[tid*J+j];
                reduction_array[tid*J+j] = 0.0;
            }
            M[j] = sqrt(lambda*freqs[j]+M[j]*subsample_rate);
            if (M[j] == 0)
                M[j] = 1;
            M[j] = 1/M[j];
            g_hat[j] = g_w[j]*M[j];
            r[j] = -g_hat[j];
            d[j] = r[j];
            r_square += r[j]*r[j];
            g_w_square += g_hat[j]*g_hat[j];
        }
    }
    else
    {
#pragma omp parallel for num_threads(nr_threads) schedule(static) reduction(+:r_square, g_w_square)
        for (AfmIdx j = 0; j < J; j++)
        {
            M[j] = 1;
            g_hat[j] = g_w[j]*M[j];
            r[j] = -g_hat[j];
            d[j] = r[j];
            r_square += r[j]*r[j];
            g_w_square += g_hat[j]*g_hat[j];
        }
    }

    AfmVal nr_cg_w = 0;
    while(1)
    {
        nr_cg_w++;
#pragma omp parallel for num_threads(nr_threads) schedule(static)
        for (AfmIdx j = 0; j < J; j++)
        {
            d[j] = r[j]+beta*d[j];
            H_w_d[j] = lambda*freqs[j]*d[j]*M[j];
        }
#pragma omp parallel for num_threads(nr_threads) schedule(guided)
        for(AfmIdx i_ = 0; i_ < data.I/subsample_rate; i_++)
        {
            AfmIdx tid = omp_get_thread_num();
            AfmIdx i = orders[i_];
            AfmVal xd = 0.0;
            for(Node *x = X[i]; x != X[i+1]; x++)
            {
                const AfmIdx j = x->idx;
                const AfmVal x_ij = x->val;
                xd += x_ij*d[j]*M[j];
            }
            const AfmVal tau1 = sigma[i]*(1-sigma[i])*xd*subsample_rate;
            for(Node *x = X[i]; x != X[i+1]; x++)
                reduction_array[tid*J + x->idx] += tau1*x->val;
        }

        AfmVal dTH_w_d = 0.0;
#pragma omp parallel for num_threads(nr_threads) schedule(static) reduction(+:dTH_w_d)
        for (AfmIdx j = 0; j < J; j++)
        {
            for(AfmIdx tid = 0 ; tid < nr_threads; tid++)
            {
                H_w_d[j] += reduction_array[tid*J+j];
                reduction_array[tid*J+j] = 0;
            }
            H_w_d[j] *= M[j];
            dTH_w_d += d[j]*H_w_d[j];
        }

        gamma = r_square;
        alpha = gamma/dTH_w_d;
        r_square = 0.0;
#pragma omp parallel for num_threads(nr_threads) schedule(static) reduction(+:r_square)
        for(AfmIdx j = 0; j < J; j++)
        {
            s_w[j] += alpha*d[j];
            r[j] -= alpha*H_w_d[j];
            r_square += r[j]*r[j];
        }
        if(g_w_square*0.3*0.3 > r_square && is_relative_cg)
        {
            cout << "nr_cg_w: " << nr_cg_w << endl;
            break;
        }
        if(nr_cg_w >= cg_iters && !is_relative_cg)
        {
            cout << "nr_cg_w: " << nr_cg_w << endl;
            break;
        }
        beta = r_square/gamma;
    }
#pragma omp parallel for num_threads(nr_threads) schedule(static)
    for (AfmIdx j = 0; j < J; j++)
       s_w[j] *= M[j];
}

void pcg_U(const AfmVal lambda,
    const AfmData &data,
    const AfmVal *U,
    const AfmVal *V,
    const AfmIdx J,
    const AfmIdx K,
    const vector<AfmVal> &sigma,
    const vector<AfmVal> &freqs,
    const bool do_pcond,
    const AfmIdx do_gd,
    const AfmIdx cg_iters,
    const vector<AfmVal> &g_U,
    const AfmVal *Q,
    const vector<AfmIdx> &orders,
    vector<AfmVal> &s_U)
{
    if (do_gd)
    {
        for (AfmIdx j = 0; j < J; j++)
            for (AfmIdx k = 0; k < K; k++)
                s_U[j*K+k] = -g_U[j*K+k];
        return;
    }
    else
    {
        for (AfmIdx j = 0; j < J; j++)
            for (AfmIdx k = 0; k < K; k++)
                s_U[j*K+k] = 0;
    }

    vector<AfmVal> M(J*K, 0);
    vector<AfmVal> d(J*K, 0);
    vector<AfmVal> r(J*K, 0);
    vector<AfmVal> reduction_array(nr_threads*J*K, 0);
    vector<AfmVal> H_U_d(J*K, 0);
    vector<AfmVal> g_U_hat(J*K, 0);
    AfmVal alpha = 0;
    AfmVal beta = 0;
    AfmVal gamma = 0;
    AfmVal r_square = 0.0;
    AfmVal g_U_square = 0.0;
    vector<Node*> X = data.indexes;
    AfmIdx nr_cg_U = 0;

    uniform_int_distribution<AfmIdx> dist(0, data.I-1);
    minstd_rand0 generator(rand());

    if (do_pcond)
    {
#pragma omp parallel for num_threads(nr_threads) schedule(guided)
        for (AfmIdx i_ = 0; i_ < data.I/subsample_rate; i_++)
        {
            AfmIdx tid = omp_get_thread_num();
            AfmIdx i = orders[i_];
            const AfmVal *q = Q+i*K;
            for (Node *node = X[i]; node!= X[i+1]; node++)
            {
                const AfmIdx j = node->idx;
                const AfmVal x = node->val;
                const AfmVal tau1 = sigma[i]*(1-sigma[i]);
                for (AfmIdx k = 0; k < K; k++)
                {
                    const AfmIdx jk = j*K+k;
                    const AfmVal xi = 0.5*q[k]*x;
                    reduction_array[tid*J*K+jk] += tau1*xi*xi;
                }
            }
        }

#pragma omp parallel for num_threads(nr_threads) schedule(static) reduction(+:r_square, g_U_square)
        for (AfmIdx jk = 0; jk < J*K; jk++)
        {
            for(AfmIdx tid = 0; tid < nr_threads; tid++)
            {
                M[jk] += reduction_array[tid*J*K+jk];
                reduction_array[tid*J*K+jk] = 0;
            }
            M[jk] = sqrt((M[jk]*subsample_rate+lambda));
            if (M[jk] == 0)
                M[jk] = 1;
            M[jk] = 1/M[jk];
            g_U_hat[jk] = g_U[jk]*M[jk];
            r[jk]  = -g_U_hat[jk];
            d[jk] = r[jk];
            r_square += r[jk]*r[jk];
            g_U_square += g_U_hat[jk]*g_U_hat[jk];
        }
    }
    else
    {
#pragma omp parallel for num_threads(nr_threads) schedule(static) reduction(+:r_square, g_U_square)
        for (AfmIdx jk = 0; jk < J*K; jk++)
        {
            M[jk] = 1;
            g_U_hat[jk] = g_U[jk]*M[jk];
            r[jk]  = -g_U_hat[jk];
            d[jk] = r[jk];
            r_square += r[jk]*r[jk];
            g_U_square += g_U_hat[jk]*g_U_hat[jk];
        }
    }

    while(1)
    {
        nr_cg_U++;

#pragma omp parallel for  num_threads(nr_threads) schedule(static)
        for (AfmIdx jk = 0; jk < J*K; jk++)
        {
            d[jk] = r[jk]+beta*d[jk];
            H_U_d[jk] = lambda*d[jk]*M[jk];
        }
#pragma omp parallel for num_threads(nr_threads) schedule(guided)
        for(AfmIdx i_ = 0; i_ < data.I/subsample_rate; i_++)
        {
            AfmIdx tid = omp_get_thread_num();
            AfmIdx i = orders[i_];
            AfmVal tau = 0.0;
            const AfmVal *q = Q+i*K;
            for(const Node *node = X[i]; node != X[i+1]; node++)
            {
                const AfmIdx j = node->idx;
                const AfmVal x = node->val;
                AfmVal tau1 = 0.0;
                for(AfmIdx k = 0; k < K; k++)
                {
                    const AfmIdx jk = j*K+k;
                    tau1 += 0.5*q[k]*x*d[jk]*M[jk];
                }
                tau += tau1;
            }
            tau = sigma[i]*(1-sigma[i])*tau*subsample_rate;
            for(Node *node = X[i]; node != X[i+1]; node++)
            {
                const AfmIdx j = node->idx;
                const AfmVal x = node->val;
                for(AfmIdx k = 0; k < K; k++)
                {
                    const AfmIdx jk = j*K+k;
                    reduction_array[tid*J*K+jk] += tau*q[k]*x*0.5;
                }
            }
        }

        AfmVal dHd = 0.0;
#pragma omp parallel for num_threads(nr_threads) schedule(static) reduction(+:dHd)
        for (AfmIdx jk = 0; jk < J*K; jk++)
        {
            for (AfmIdx tid = 0; tid < nr_threads; tid++)
            {
                    H_U_d[jk] += reduction_array[tid*J*K+jk];
                    reduction_array[tid*J*K+jk] = 0;
            }
                H_U_d[jk] *= M[jk];
                dHd += d[jk]*H_U_d[jk];
        }

        gamma = r_square;
        alpha = gamma/dHd;
        r_square = 0.0;
#pragma omp parallel for num_threads(nr_threads) schedule(static) reduction(+:r_square)
        for (AfmIdx jk = 0; jk < J*K; jk++)
        {
            s_U[jk] += alpha*d[jk];
            r[jk] -= alpha*H_U_d[jk];
            r_square += r[jk]*r[jk];
        }
        if(g_U_square*0.3*0.3 > r_square && is_relative_cg)
        {
            //cout << "nr_cg_U: " << nr_cg_U << endl;
            break;
        }
        if(nr_cg_U >= cg_iters && !is_relative_cg)
        {
            //cout << "nr_cg_U: " << nr_cg_U << endl;
            break;
        }
        beta = r_square/gamma;
    }
#pragma omp parallel for num_threads(nr_threads) schedule(static)
    for (AfmIdx jk = 0; jk < J*K; jk++)
    {
        s_U[jk] *= M[jk];
    }
}

AfmVal fm_predict(const Node *begin, const Node *end,
    const AfmModel &model, const bool drop_self)
{
    const AfmIdx K = model.K;
    const AfmVal *U = model.U.data();
    AfmVal z = 0;
    vector<AfmVal> p(K, 0);
    for (const Node *x = begin; x != end; x++)
    {
        if (x->idx > model.J-1)
            continue;
        AfmVal uu = 0;
        z += model.w[x->idx]*x->val; // linear term
        const AfmVal *u = U+x->idx*K;
        for (AfmIdx k = 0; k < K; k++)
        {
            p[k] += u[k]*x->val;
            uu += u[k]*u[k];
        }
        if (drop_self)
            z -= 0.5*uu*x->val*x->val;
    }
    for (AfmIdx k = 0; k < K; k++)
        z += 0.5*p[k]*p[k];
    return z;
}

AfmVal afm_predict(const Node *begin, const Node *end,
    const AfmModel &model, const bool drop_self)
{
    const AfmIdx K = model.K;
    const AfmVal *U = model.U.data();
    const AfmVal *V = model.V.data();
    vector<AfmVal> p(K, 0);
    vector<AfmVal> q(K, 0);
    AfmVal z = 0;
    for (const Node *x = begin; x != end; x++)
    {
        if(x->idx > model.J-1)
            continue;
        z += model.w[x->idx]*x->val; // linear term
        const AfmVal *u = U+x->idx*K;
        const AfmVal *v = V+x->idx*K;
        AfmVal uv = 0;
        for (AfmIdx k = 0; k < K; k++)
        {
            p[k] += u[k]*x->val;
            q[k] += v[k]*x->val;
            uv += u[k]*v[k];
        }
        if (drop_self)
            z -= 0.5*uv*x->val*x->val;
    }
    for (AfmIdx k = 0; k < K; k++)
        z += 0.5*p[k]*q[k];
    return z;
}

AfmVal evaluate(const AfmData &data,
    const AfmModel &model,
    const bool drop_self)
{
    AfmVal loss = 0;
#pragma omp parallel for num_threads(nr_threads) schedule(guided) reduction(+:loss)
    for (AfmIdx i = 0; i < data.I; i++)
    {
        AfmVal z = 0;
        if (model.solver == AFM_ADAGRAD ||
            model.solver == AFM_ALS_PRIMAL)
            z = afm_predict(data.indexes[i], data.indexes[i+1], model, drop_self);
        else if(model.solver == FM_ADAGRAD)
            z = fm_predict(data.indexes[i], data.indexes[i+1], model, drop_self);
        else if(model.solver == FM_AGD ||
                model.solver == FM_GD)
            z = fm_predict(data.indexes[i], data.indexes[i+1], model, true);
        const AfmVal yz = data.labels[i]*z;
        if (yz > 0)
            loss += log(1.0+exp(-yz));
        else
            loss += -yz+log(1.0+exp(yz));
    }
    return loss/data.I;
}

AfmVal accuracy(const AfmData &data,
    const AfmModel &model,
    const bool drop_self)
{
    AfmVal correct = 0;
    for (AfmIdx i = 0; i < data.I; i++)
    {
        AfmVal z = 0;
        if (model.solver == AFM_ADAGRAD ||
            model.solver == AFM_ALS_PRIMAL)
            z = afm_predict(data.indexes[i], data.indexes[i+1], model, drop_self);
        else if(model.solver == FM_ADAGRAD)
            z = fm_predict(data.indexes[i], data.indexes[i+1], model, drop_self);
        else if(model.solver == FM_AGD ||
                model.solver == FM_GD)
            z = fm_predict(data.indexes[i], data.indexes[i+1], model, true);
        const AfmVal yz = data.labels[i]*z;
        if (yz > 0)
            correct += 1;
    }
    return correct/data.I;
}

AfmVal calc_norm(const AfmVal *vec, const AfmIdx len)
{
    AfmVal norm = 0;
    for (AfmIdx j = 0; j < len; j++)
        norm += vec[j]*vec[j];
    return sqrt(norm);
}

AfmVal calc_linear_term(const AfmVal *w,
    const Node *begin, const Node *end)
{
    AfmVal z = 0;
    for (const Node *node = begin; node != end; node++)
    {
        const AfmIdx j = node->idx;
        const AfmVal x = node->val;
        z += w[j]*x;
    }
    return z;
}

AfmVal afm_calc_interaction_no_self(
    const AfmVal *U, const AfmVal *V,
    const AfmIdx K, const Node *begin,
    const Node *end, vector<AfmVal> &p,
    vector<AfmVal> &q)
{
    AfmVal z = 0;
    for (const Node *node = begin; node != end; node++)
    {
        const AfmIdx j = node->idx;
        const AfmVal x_ij = node->val;
        AfmVal uv = 0;
        for (AfmIdx k = 0; k < K; k++)
        {
            const AfmIdx jk = j*K+k;
            p[k] += U[jk]*x_ij;
            q[k] += V[jk]*x_ij;
            uv += U[jk]*V[jk];
        }
        z -= 0.5*uv*x_ij*x_ij;
    }
    for (AfmIdx k = 0; k < K; k++)
        z += 0.5*p[k]*q[k];
    return z;
}

AfmVal afm_calc_interaction_self(
    const AfmVal *U, const AfmVal *V,
    const AfmIdx K, const Node *begin,
    const Node *end, vector<AfmVal> &p,
    vector<AfmVal> &q)
{
    AfmVal z = 0;
    for (const Node *node = begin; node != end; node++)
    {
        const AfmIdx j = node->idx;
        const AfmVal x_ij = node->val;
        for (AfmIdx k = 0; k < K; k++)
        {
            const AfmIdx jk = j*K+k;
            p[k] += U[jk]*x_ij;
            q[k] += V[jk]*x_ij;
        }
    }
    for (AfmIdx k = 0; k < K; k++)
        z += 0.5*p[k]*q[k];
    return z;
}

inline AfmVal afm_calc_interaction(
    const AfmIdx K,
    const AfmVal *p,
    const AfmVal *q)
{
    AfmVal z = 0;
    for (AfmIdx k = 0; k < K; k++)
        z += 0.5*p[k]*q[k];
    return z;
}

void als_train_afm_primal(
    const AfmData &data,
    const AfmData &test,
    const Parameter param,
    const bool use_freq,
    const bool do_pcond,
    const bool do_gd,
    AfmModel &model)
{
    if (data.J != model.J)
        return;
    const AfmIdx T = param.iters;
    const AfmIdx I = data.I;
    const AfmIdx J = data.J;
    const AfmIdx K = param.K;
    const AfmIdx cg_iters = param.cg_iters;
    const vector<AfmVal> &Y = data.labels;
    const vector<Node*> &X = data.indexes;
    const AfmVal nu = param.nu;
    const AfmVal kappa = param.kappa;
    const AfmVal tol = param.tol;
    AfmIdx nr_searches_w = 0;
    AfmIdx nr_searches_U = 0;
    AfmIdx nr_searches_V = 0;

    vector<AfmVal> P_(I*K, 0);
    vector<AfmVal> Q_(I*K, 0);
    vector<AfmVal> freqs(J, 0);
    vector<AfmIdx> orders(I, 0);

    if (use_freq)
    {
        for (AfmIdx i = 0; i < I; i++)
            for (Node *x = X[i]; x!= X[i+1]; x++)
                freqs[x->idx]++;
        for (AfmIdx j = 0; j < J; j++)
            if (freqs[j] == 0)
                freqs[j] = 1;
    }
    else
    {
        for (AfmIdx j = 0; j < J; j++)
            freqs[j] = 1;
    }

    auto update_U = [&] (const AfmVal t, const AfmVal lambda_U,
        const AfmVal *w, const AfmVal *V,
        AfmIdx &nr_searches_U, AfmVal &norm_g_U,
        AfmVal &norm_g_U_0, AfmVal *P, AfmVal *Q,
        AfmVal *U) -> void
    {
        //////////////////////////////////////////////////
        // Part 1: compute objective function and gradient
        //////////////////////////////////////////////////
        AfmVal F_U = 0;
        vector<AfmVal> g_U(K*J, 0);
        vector<AfmVal> g_U_tmp(nr_threads*K*J, 0.0);
        vector<AfmVal> s_U(K*J, 0);
        vector<AfmVal> sigma(I, 0);
#pragma omp parallel for num_threads(nr_threads) schedule(static) reduction(+:F_U)
        for (AfmIdx jk = 0; jk < J*K; jk++)
        {
            F_U += lambda_U*U[jk]*U[jk];
            g_U[jk] = lambda_U*U[jk];
        }
        F_U *= 0.5;
#pragma omp parallel for num_threads(nr_threads) schedule(guided) reduction(+:F_U)
        for (AfmIdx i = 0; i < I; i++)
        {
            AfmIdx tid = omp_get_thread_num();
            AfmVal *p = P+i*K;
            AfmVal *q = Q+i*K;
            AfmVal z = 0;
            for (Node *node = X[i]; node != X[i+1]; node++)
            {
                const AfmIdx j = node->idx;
                const AfmVal x = node->val;
                z += w[j]*x;
            }
            for (AfmIdx k = 0; k < K; k++)
                z += 0.5*p[k]*q[k];
            const AfmVal yz = Y[i]*z;
            if (yz > 0)
            {
                const AfmVal exp_m = exp(-yz);
                sigma[i] = exp_m/(1+exp_m);
                F_U += log(1+exp_m);
            }
            else
            {
                const AfmVal exp_m = exp(yz);
                sigma[i] = 1/(1+exp_m);
                F_U += -yz+log(1+exp_m);
            }
            for (Node *node = X[i]; node != X[i+1]; node++)
            {
                const AfmIdx j = node->idx;
                const AfmVal x = node->val;
                for (AfmIdx k = 0; k < K; k++)
                {
                    const AfmIdx jk = j*K+k;
                    g_U_tmp[tid*J*K+jk] -= Y[i]*sigma[i]*q[k]*x*0.5;
                }
            }
        }

#pragma omp parallel for num_threads(nr_threads) schedule(static)
        for (AfmIdx jk = 0; jk < J*K; jk++)
        {
            for(AfmIdx tid = 0; tid < nr_threads; tid++)
            {
                g_U[jk] += g_U_tmp[tid*J*K+jk];
            }
        }

        // compute gradient norm
        norm_g_U = calc_norm(g_U.data(), J*K);
        // store initial gradient norm
        if (norm_g_U_0 <= 0)
            norm_g_U_0 = norm_g_U;

        if(nt_eps*norm_g_U_0 > norm_g_U)
            return;

        if(norm_g_U < 1e-5 && t > 2)
        {
            norm_g_U = 0;
            return;
        }

        //////////////////////////////////////////////////
        // Part 2: compute descent direction and update
        //////////////////////////////////////////////////
        if (norm_g_U > tol*norm_g_U_0)
            pcg_U(lambda_U, data, U, V, J, K, sigma, freqs,
                  do_pcond, do_gd, cg_iters, g_U, Q, orders, s_U);

        //////////////////////////////////////////////////
        // Part 3: line search and update
        //////////////////////////////////////////////////
        if (norm_g_U > tol*norm_g_U_0)
        {
            AfmVal phi_U = 0;
            AfmVal F_U_theta = 0;
            AfmVal theta_U = 1;
#pragma omp parallel for num_threads(nr_threads) schedule(static) reduction(+:F_U_theta, phi_U)
            for (AfmIdx jk = 0; jk < J*K; jk++)
            {
                // compute required descent
                phi_U += -g_U[jk]*s_U[jk];
                // update model
                U[jk] += theta_U*s_U[jk];
                // accumulate regularization part
                F_U_theta += lambda_U*U[jk]*U[jk];
            }
            F_U_theta *= 0.5;
            phi_U *= nu;
#pragma omp parallel for num_threads(nr_threads) schedule(guided) reduction(+:F_U_theta)
            for (AfmIdx i = 0; i < I; i++)
            {
                // update cached p_i and q_i and
                // compute prediction value
                AfmVal *p = P+i*K;
                AfmVal *q = Q+i*K;
                for (AfmIdx k = 0; k < K; k++)
                    p[k] = 0;
                AfmVal z = 0;
                for (Node *node = X[i]; node != X[i+1]; node++)
                {
                    const AfmIdx j = node->idx;
                    const AfmVal x = node->val;
                    z += w[j]*x;
                    for (AfmIdx k = 0; k < K; k++)
                    {
                        const AfmIdx jk = j*K+k;
                        p[k] += U[jk]*x;
                    }
                }
                for (AfmIdx k = 0; k < K; k++)
                    z += 0.5*p[k]*q[k];
                // accumulate loss
                const AfmVal yz = Y[i]*z;
                if (yz > 0)
                    F_U_theta += log(1.0+exp(-yz));
                else
                    F_U_theta += -yz+log(1.0+exp(yz));
            }
            nr_searches_U = 0;
            // do line search
            while (F_U-F_U_theta < theta_U*phi_U)
            {
                if (nr_searches_U > 30)
                    return;
                nr_searches_U++;
                // decrease step size
                theta_U *= kappa;
                // compute new objective value
                F_U_theta = 0;
                const AfmVal theta_U1 = (1-kappa)/kappa*theta_U;
#pragma omp parallel for num_threads(nr_threads) schedule(guided) reduction(+:F_U_theta)
                for (AfmIdx jk = 0; jk < J*K; jk++)
                {
                    // update model
                    U[jk] -= theta_U1*s_U[jk];
                    // accumulate regularization
                    F_U_theta += lambda_U*U[jk]*U[jk];
                }
                F_U_theta *= 0.5;
#pragma omp parallel for num_threads(nr_threads) schedule(guided) reduction(+:F_U_theta)
                for (AfmIdx i = 0; i < I; i++)
                {
                    // update cached p_i and
                    // compute prediction value
                    AfmVal *p = P+i*K;
                    AfmVal *q = Q+i*K;
                    AfmVal z = 0;
                    for (AfmIdx k = 0; k < K; k++)
                        p[k] = 0;
                    for (Node *node = X[i]; node != X[i+1]; node++)
                    {
                        const AfmIdx j = node->idx;
                        const AfmVal x = node->val;
                        z += w[j]*x;
                        for (AfmIdx k = 0; k < K; k++)
                        {
                            const AfmIdx jk = j*K+k;
                            p[k] += U[jk]*x;
                        }
                    }
                    for (AfmIdx k = 0; k < K; k++)
                        z += 0.5*p[k]*q[k];

                    // accumulate loss
                    const AfmVal yz = Y[i]*z;
                    if (yz > 0)
                        F_U_theta += log(1.0+exp(-yz));
                    else
                        F_U_theta += -yz+log(1.0+exp(yz));

                }
            }
        }
    };

    auto update_w = [&](const AfmIdx t, const AfmVal lambda_w,
        const AfmVal *U, const AfmVal *V,
        AfmIdx &nr_searches_w, AfmVal &norm_g_w,
        AfmVal &norm_g_w_0, AfmVal *P, AfmVal *Q, AfmVal *w) -> void
    {
        //////////////////////////////////////////////////
        // Part 1: compute objective function and gradient
        //////////////////////////////////////////////////
        vector<AfmVal> s_w(J, 0);
        vector<AfmVal> g_w(J, 0);
        vector<AfmVal> g_w_tmp(nr_threads * J);
        vector<AfmVal> c(I, 0);
        vector<AfmVal> sigma(I, 0);
        AfmVal F_w = 0;
#pragma omp parallel for num_threads(nr_threads) schedule(static) reduction(+:F_w)
        for (AfmIdx j = 0; j < J; j++)
        {
            const AfmVal lambda_w_j = lambda_w*freqs[j];
            // regularization part in function value
            F_w += lambda_w_j*w[j]*w[j];
            // regularization part in gradient
            g_w[j] = lambda_w_j*w[j];
            for(AfmIdx tid =0; tid < nr_threads; tid++)
            {
                g_w_tmp[tid*J+j] = 0.0;
            }
        }
        F_w *= 0.5;
#pragma omp parallel for num_threads(nr_threads) schedule(guided) reduction(+:F_w)
        for (AfmIdx i = 0; i < I; i++)
        {
            AfmIdx tid = omp_get_thread_num();
            const AfmVal *p = P+i*K;
            const AfmVal *q = Q+i*K;
            c[i] = 0;
            for (AfmIdx k = 0; k < K; k++)
                c[i] += 0.5*p[k]*q[k];
            AfmVal yz = Y[i]*(c[i]+calc_linear_term(w, X[i], X[i+1]));

            // accumulate loss
            if (yz > 0)
            {
                const AfmVal exp_m = exp(-yz);
                sigma[i] = exp_m/(1+exp_m);
                F_w += log(1+exp_m);
            }
            else
            {
                const AfmVal exp_m = exp(yz);
                sigma[i] = 1/(1+exp_m);
                F_w += -yz+log(1+exp_m);
            }
            for (Node *x = X[i]; x != X[i+1]; x++)
                g_w_tmp[tid*J+x->idx] -= Y[i]*sigma[i]*x->val;
        }

#pragma omp parallel for num_threads(nr_threads) schedule(static)
        for(AfmIdx j = 0; j < J; j++)
        {
            for(AfmIdx tid = 0; tid < nr_threads; tid++)
            {
                g_w[j] += g_w_tmp[tid*J+j];
            }
        }

        // compute current gradient norm
        norm_g_w = calc_norm(g_w.data(), J);
        // store initial gradient norm
        if (norm_g_w_0 <= 0)
            norm_g_w_0 = norm_g_w;

        if(nt_eps*norm_g_w_0 > norm_g_w)
            return;

        if(norm_g_w < 1e-5 && t > 2)
        {
            norm_g_w = 0;
            return;
        }

        //////////////////////////////////////////////////
        // Part 2: compute descent direction and update
        //////////////////////////////////////////////////
        if (norm_g_w > tol*norm_g_w_0)
            pcg_w(lambda_w, data, sigma, freqs,
                  do_pcond, do_gd, cg_iters, g_w, orders, s_w);

        //////////////////////////////////////////////////
        // Part 3: line search and update
        //////////////////////////////////////////////////
        if (norm_g_w > tol*norm_g_w_0)
        {
            // do line search
            AfmVal phi_w = 0;
            AfmVal F_w_theta = 0;
            AfmVal theta_w = 1;
#pragma omp parallel for num_threads(nr_threads) schedule(static) reduction(+:F_w_theta, phi_w)
            for (AfmIdx j = 0; j < J; j++)
            {
                phi_w -= g_w[j]*s_w[j];
                w[j] += theta_w*s_w[j];
                F_w_theta += lambda_w*freqs[j]*w[j]*w[j];
            }
            F_w_theta *= 0.5;
            phi_w *= nu;
#pragma omp parallel for num_threads(nr_threads) schedule(guided) reduction(+:F_w_theta)
            for (AfmIdx i = 0; i < I; i++)
            {
                const AfmVal yz = Y[i]*(c[i]+calc_linear_term(w, X[i], X[i+1]));
                if (yz > 0)
                    F_w_theta += log(1.0+exp(-yz));
                else
                    F_w_theta += -yz+log(1.0+exp(yz));
            }
            nr_searches_w = 0;
            while (F_w-F_w_theta < theta_w*phi_w)
            {
                if (nr_searches_w > 30)
                    return;
                nr_searches_w++;
                F_w_theta = 0;
                theta_w *= kappa;
                const AfmVal theta_w1 = (1-kappa)/kappa*theta_w;
#pragma omp parallel for num_threads(nr_threads) schedule(static) reduction(+:F_w_theta)
                for (AfmIdx j = 0; j < J; j++)
                {
                    w[j] -= theta_w1*s_w[j];
                    F_w_theta += lambda_w*freqs[j]*w[j]*w[j];
                }
                F_w_theta *= 0.5;
#pragma omp parallel for num_threads(nr_threads) schedule(guided) reduction(+:F_w_theta)
                for (AfmIdx i = 0; i < I; i++)
                {
                    const AfmVal yz = Y[i]*(c[i]+calc_linear_term(w, X[i], X[i+1]));
                    if (yz > 0)
                        F_w_theta += log(1.0+exp(-yz));
                    else
                        F_w_theta += -yz+log(1.0+exp(yz));
                }
            }
        }
    };


#pragma omp parallel for num_threads(nr_threads) schedule(guided)
    for (AfmIdx i = 0; i < I; i++)
    {
        orders[i] = i;
        const AfmVal *U = model.U.data();
        const AfmVal *V = model.V.data();
        AfmVal *P = P_.data();
        AfmVal *Q = Q_.data();
        for (const Node *node = X[i]; node != X[i+1]; node++)
        {
            const AfmIdx j = node->idx;
            const AfmVal x = node->val;
            for (AfmIdx k = 0; k < K; k++)
            {
                const AfmIdx ik = i*K+k;
                const AfmIdx jk = j*K+k;
                P[ik] += U[jk]*x;
                Q[ik] += V[jk]*x;
            }
        }
    }

    double total_time = 0;
    for (AfmIdx t = 0; t < T ; t++)
    {
        AfmVal norm_g_w_0 = 0;
        AfmVal norm_g_U_0 = 0;
        AfmVal norm_g_V_0 = 0;
        AfmVal norm_g_w = 0;
        AfmVal norm_g_U = 0;
        AfmVal norm_g_V = 0;
        random_shuffle(orders.begin(), orders.end());
        auto start_time = omp_get_wtime();
        
        AfmIdx nt_count = 0;
        while (param.lambda_w >= 0 && nt_eps*norm_g_w_0 <= norm_g_w && nt_count < max_nt_iters)
        {
            update_w(t, param.lambda_w, model.U.data(), model.V.data(),
                     nr_searches_w, norm_g_w, norm_g_w_0,
                     P_.data(), Q_.data(), model.w.data());
            if (nr_searches_w > 30)
                break;
            nt_count ++;
        }

        nt_count = 0;
        while (param.lambda_U >= 0 && nt_eps*norm_g_U_0 <= norm_g_U && nt_count < max_nt_iters)
        {
            update_U(t, param.lambda_U, model.w.data(), model.V.data(),
                nr_searches_U, norm_g_U, norm_g_U_0,
                P_.data(), Q_.data(), model.U.data());
            if (nr_searches_U > 30)
                break;
            nt_count ++;
        }

        nt_count = 0;
        while (param.lambda_V >= 0 && nt_eps*norm_g_V_0 <= norm_g_V && nt_count < max_nt_iters)
        {
            update_U(t, param.lambda_V, model.w.data(), model.U.data(),
                nr_searches_V, norm_g_V, norm_g_V_0,
                Q_.data(), P_.data(), model.V.data());
            if (nr_searches_V > 30)
                break;
            nt_count ++;
        }

        auto iter_time = omp_get_wtime()-start_time;
        total_time += iter_time;

        ///////////////////////////////////////////////////
        // Part 3: Output information
        ///////////////////////////////////////////////////
        const AfmVal *w = model.w.data();
        const AfmVal *U = model.U.data();
        const AfmVal *V = model.V.data();
        AfmVal norm_w = 0;
        AfmVal norm_U = 0;
        AfmVal norm_V = 0;
        AfmVal loss = 0;

#pragma omp parallel for num_threads(nr_threads) schedule(static) reduction(+:norm_w, norm_V, norm_U)
        for (AfmIdx j = 0; j < J; j++)
        {
            const AfmVal lambda_U_j = param.lambda_U*freqs[j];
            const AfmVal lambda_V_j = param.lambda_V*freqs[j];
            if (param.lambda_w > 0)
                norm_w += param.lambda_w*freqs[j]*w[j]*w[j];
            if (param.lambda_U > 0)
            {
                for (AfmIdx k = 0; k < K; k++)
                {
                    const AfmIdx jk = j*K+k;
                    norm_U += lambda_U_j*U[jk]*U[jk];
                    norm_V += lambda_V_j*V[jk]*V[jk];
                }
            }
        }

#pragma omp parallel for num_threads(nr_threads) schedule(guided) reduction(+:loss)
        for (AfmIdx i = 0; i < I; i++)
        {
            vector<AfmVal> p(K, 0);
            vector<AfmVal> q(K, 0);
            const AfmVal yz = Y[i]*(calc_linear_term(w, X[i], X[i+1])+
                             afm_calc_interaction_self(U, V, K, X[i], X[i+1], p, q));
            if (yz > 0)
                loss += log(1.0+exp(-yz));
            else
                loss += -yz+log(1.0+exp(yz));
        }

        AfmVal obj = 0.5*(norm_w+norm_U+norm_V)+loss;
        AfmVal loss_test = test.I <= 0? -1: evaluate(test, model, false);
        AfmVal acc_test = test.I <= 0? -1: accuracy(test, model, false);
        cout << "iter: " << std::setfill(' ') << std::setw(4) << t
             << " ls(w): " << std::setw(2) << nr_searches_w
             << " ls(U): " << std::setw(2) << nr_searches_U
             << " ls(V): " << std::setw(2) << nr_searches_V
             << " time: " << std::setw(9) << std::setprecision(8) << total_time
             << " obj: " << std::setw(9) << std::setprecision(8) << obj
             << " |g_w|: " << std::setw(9) << std::setprecision(8) << norm_g_w
             << " |g_U|: " << std::setw(9) << std::setprecision(8) << norm_g_U
             << " |g_V|: " << std::setw(9) << std::setprecision(8) << norm_g_V
             << " tr_loss: " << std::setw(9) << std::setprecision(8) << loss/data.I
             << " te_loss: " << std::setw(9) << std::setprecision(8) << loss_test
             << " te_acc: " << std::setw(9) << std::setprecision(8) << acc_test
             << endl;
    }
}

void train_afm_adagrad(
    const AfmData &data,
    const AfmData &data_test,
    const Parameter param,
    const bool use_freq,
    AfmModel &model)
{
    const AfmIdx T = param.iters;
    const AfmIdx J = data.J;
    const AfmIdx K = param.K;
    const AfmVal lambda_w = param.lambda_w;
    const AfmVal lambda_U = param.lambda_U;
    const AfmVal lambda_V = param.lambda_V;
    const vector<AfmVal> &Y = data.labels;
    const vector<Node*> &X = data.indexes;
    const AfmVal eta = param.eta;
    AfmVal *w = model.w.data();
    AfmVal *U = model.U.data();
    AfmVal *V = model.V.data();
    vector<AfmVal> G_w(J, 1);
    vector<AfmVal> G_U(J*K, 1);
    vector<AfmVal> G_V(J*K, 1);
    vector<AfmIdx> ins_indexes(data.I);
    vector<AfmVal> freqs(J, 0);
    AfmVal best_logloss = 1e10;

    for (AfmIdx i = 0; i < data.I; i++)
    {
        ins_indexes[i] = i;
        for (Node *x = X[i]; x != X[i+1]; ++x)
            freqs[x->idx]++;
    }
    for (AfmIdx j = 0; j < J; j++)
        if (freqs[j] == 0)
            freqs[j] = 1;

    cout.width(4);
    cout << "iter";
    cout.width(13);
    cout << "tr_logloss";
    cout.width(13);
    cout << "va_logloss";
    cout.width(13);
    cout << "time";
    cout << endl;

    double total_time = 0;
    for (AfmIdx t = 0; t < T; t++)
    {
        auto start_time = omp_get_wtime();
        random_shuffle(ins_indexes.begin(), ins_indexes.end());
        AfmVal loss = 0;
#pragma omp parallel for num_threads(nr_threads) schedule(guided) reduction(+:loss)
        for (AfmIdx i_ = 0; i_ < data.I; i_++)
        {
            const AfmIdx i = ins_indexes[i_];
            vector<AfmVal> p(K,0);
            vector<AfmVal> q(K,0);
            const AfmVal z =
                calc_linear_term(w, X[i], X[i+1])+
                afm_calc_interaction_self(U, V, K, X[i], X[i+1], p, q);

            const AfmVal sigma = -Y[i]/(1.0+exp(Y[i]*z));
            const AfmVal yz = Y[i]*z;
            if (yz > 0)
                loss += log(1.0+exp(-yz));
            else
                loss += -yz+log(1.0+exp(yz));
            for (Node *node = X[i]; node != X[i+1]; node++)
            {
                const AfmVal x = node->val;
                const AfmIdx j = node->idx;
                if (lambda_w >= 0)
                {
                    const AfmVal lambda_w_j =
                        use_freq? lambda_w: lambda_w/freqs[j];
                    const AfmVal g_w = lambda_w_j*w[j]+sigma*x;
                    G_w[j] += g_w*g_w;
                    w[j] -= eta/sqrt(G_w[j])*g_w;
                }
                if (lambda_U >= 0)
                {
                    const AfmVal lambda_U_j =
                        use_freq? lambda_U: lambda_U/freqs[j];
                    for (AfmIdx k = 0; k < K; k++)
                    {
                        const AfmIdx jk = j*K+k;
                        const AfmVal g_u = lambda_U_j*U[jk]+
                            0.5*sigma*q[k]*x;
                            
                        G_U[jk] += g_u*g_u;
                        U[jk] -= eta/sqrt(G_U[jk])*g_u;
                    }
                }
                if (lambda_V >= 0)
                {
                    const AfmVal lambda_V_j =
                        use_freq? lambda_V: lambda_V/freqs[j];
                    for (AfmIdx k = 0; k < K; k++)
                    {
                        const AfmIdx jk = j*K+k;
                        const AfmVal g_v = lambda_V_j*V[jk]+
                            0.5*sigma*p[k]*x;
                        G_V[jk] += g_v*g_v;
                        V[jk] -= eta/sqrt(G_V[jk])*g_v;
                    }
                }
            }
        }

        auto iter_time = omp_get_wtime()-start_time;
        total_time += iter_time;

        AfmVal loss_test = data_test.I <= 0? -1: evaluate(data_test, model, false);
        if (loss_test < best_logloss)
            best_logloss = loss_test;
        else
            exit(1);
        cout.width(4);
        cout << t+1;
        cout.width(13);
        cout << std::setprecision(8) << loss/data.I;
        cout.width(13);
        cout << std::setprecision(8) << loss_test;
        cout.width(13);
        cout << std::setprecision(8) << total_time;
        cout << endl;
        }
}

AfmVal fm_calc_interaction_no_self(const AfmVal *U, const AfmIdx K,
    const Node *begin, const Node *end, vector<AfmVal> &p)
{
    AfmVal z = 0;
    for (const Node *node = begin; node != end; node++)
    {
        const AfmIdx j = node->idx;
        const AfmVal x = node->val;
        AfmVal uu = 0;
        for (AfmIdx k = 0; k < K; k++)
        {
            const AfmIdx jk = j*K+k;
            p[k] += U[jk]*x;
            uu += U[jk]*U[jk];
        }
        z -= uu*x*x;
    }
    for (AfmIdx k = 0; k < K; k++)
        z += p[k]*p[k];
    return 0.5*z;
}

AfmVal fm_calc_interaction_self(const AfmVal *U, const AfmIdx K,
    const Node *begin, const Node *end, vector<AfmVal> &p)
{
    AfmVal z = 0;
    for (const Node *node = begin; node != end; node++)
    {
        const AfmIdx j = node->idx;
        const AfmVal x = node->val;
        for (AfmIdx k = 0; k < K; k++)
            p[k] += U[j*K+k]*x;
    }
    for (AfmIdx k = 0; k < K; k++)
        z += p[k]*p[k];
    return 0.5*z;
}

void train_fm_adagrad(const AfmData &data, const AfmData &data_test,
    const Parameter param, const bool use_freq, const bool drop_self,
    AfmModel &model)
{
    const AfmIdx T = param.iters;
    const AfmIdx J = data.J;
    const AfmIdx K = param.K;
    const AfmVal lambda_w = param.lambda_w;
    const AfmVal lambda_U = param.lambda_U;
    const AfmVal *Y = data.labels.data();
    const Node* const *X = data.indexes.data();
    const AfmVal eta = param.eta;
    AfmVal *w = model.w.data();
    AfmVal *U = model.U.data();
    vector<AfmVal> G_w(J, 1);
    vector<AfmVal> G_U(J*K, 1);
    vector<AfmIdx> ins_indexes(data.I);
    vector<AfmVal> freqs(J, 0);

    for (AfmIdx i = 0; i < data.I; i++)
    {
        ins_indexes[i] = i;
        for (const Node *x = X[i]; x != X[i+1]; ++x)
            freqs[x->idx]++;
    }
    for (AfmIdx j = 0; j < J; j++)
        if (freqs[j] == 0)
            freqs[j] = 1;

    auto calc_linear_term = [] (const AfmVal *w, const Node *begin,
        const Node *end) -> AfmVal
    {
        AfmVal z = 0;
        for (const Node *node = begin; node != end; node++)
            z += w[node->idx]*node->val;
        return z;
    };

    auto calc_interaction = drop_self? fm_calc_interaction_no_self:
                                        fm_calc_interaction_self;

    auto calc_xi_no_self = [] (const AfmVal &q_k,
        const AfmVal &v_jk, const AfmVal &x_ij)
        -> AfmVal { return (q_k-v_jk*x_ij)*x_ij; };

    auto calc_xi_self = [] (const AfmVal &q_k,
        const AfmVal &v_jk, const AfmVal &x_ij)
        -> AfmVal { return q_k*x_ij; };

    auto calc_xi = drop_self? calc_xi_no_self: calc_xi_self;

    double total_time = 0;
    for (AfmIdx t = 0; t < T; t++)
    {
        auto start_time = omp_get_wtime();
        random_shuffle(ins_indexes.begin(), ins_indexes.end());
        AfmVal loss = 0;
#pragma omp parallel for num_threads(nr_threads) schedule(guided) reduction(+:loss)
        for (AfmIdx i_ = 0; i_ < data.I; i_++)
        {
            const AfmIdx i = ins_indexes[i_];
            vector<AfmVal> p(K, 0);
            const AfmVal z = calc_linear_term(w, X[i], X[i+1])+
                calc_interaction(U, K, X[i], X[i+1], p);
            const AfmVal sigma = -Y[i]/(1.0+exp(Y[i]*z));
            const AfmVal yz = Y[i]*z;
            if (yz > 0)
                loss += log(1.0+exp(-yz));
            else
                loss += -yz+log(1.0+exp(yz));
            for (const Node *node = X[i]; node != X[i+1]; node++)
            {
                const AfmIdx j = node->idx;
                const AfmVal x = node->val;
                // update w
                if (lambda_w >= 0)
                {
                    const AfmVal lambda_w_j = use_freq? lambda_w: lambda_w/freqs[j];
                    const AfmVal g_w = lambda_w_j*w[j]+sigma*x;
                    G_w[j] += g_w*g_w;
                    w[j] -= eta/sqrt(G_w[j])*g_w;
                }
                // update U
                if (lambda_U >= 0)
                {
                    const AfmVal lambda_U_j = use_freq? lambda_U: lambda_U/freqs[j];
                    for (AfmIdx k = 0; k < K; k++)
                    {
                        const AfmIdx jk = j*K+k;
                        const AfmVal g_U = lambda_U_j*U[jk]+
                            sigma*calc_xi(p[k], U[jk], x);
                        G_U[jk] += g_U*g_U;
                        U[jk] -= eta/sqrt(G_U[jk])*g_U;
                    }
                }
            }
        }

        auto iter_time = omp_get_wtime()-start_time;
        total_time += iter_time;

        AfmVal loss_test = data_test.I <= 0? -1: evaluate(data_test, model, true);
        cout.width(4);
        cout << t+1;
        cout.width(13);
        cout << std::setprecision(8) << loss/data.I;
        cout.width(13);
        cout << std::setprecision(8) << loss_test;
        cout.width(13);
        cout << std::setprecision(8) << total_time;
        cout << endl;
    }
}

void train_fm_gd(AfmData &data, AfmData &data_test,
    Parameter param, AfmModel &model)
{
    if (data.J != model.J)
        return;
    AfmVal lambda_w = param.lambda_w;
    AfmVal lambda_U = param.lambda_U;
    AfmIdx T = param.iters;
    AfmIdx I = data.I;
    AfmIdx J = data.J;
    AfmIdx K = param.K;
    vector<AfmVal> &Y = data.labels;
    vector<Node*> &X = data.indexes;
    AfmVal *w = model.w.data();
    AfmVal *U = model.U.data();
    AfmVal F = 0;
    vector<AfmVal> g_w(J, 0);
    vector<AfmVal> g_U(K*J, 0);
    vector<AfmVal> s_w(J, 0);
    vector<AfmVal> s_U(K*J, 0);
    vector<AfmVal> sigma(I, 0);
    AfmVal nu = param.nu;
    AfmVal kappa = param.kappa;
    AfmIdx nr_searches = 0;
    AfmVal tol = param.tol;
    AfmVal norm_g_0 = -1;
    AfmVal norm_g = 1e50;

    double total_time = 0;
    for (AfmIdx t = 0; t < T ; t++)
    {
        auto start_time = omp_get_wtime();
        if (norm_g <= tol*norm_g_0)
            break;

        // 1-1. Compuate objective and gradient about regularization
        F = 0;
        for (AfmIdx j = 0; j < J; j++)
        {
            F += lambda_w*w[j]*w[j];
            g_w[j] = lambda_w*w[j];
            for (AfmIdx k = 0; k < K; k++)
            {
                const AfmIdx jk = j*K+k;
                F += lambda_U*U[jk]*U[jk];
                g_U[jk] = lambda_U*U[jk];
            }
        }
        F *= 0.5;

        // 1-2. Compute objective and gradient about loss
        for (AfmIdx i = 0; i < I; i++)
        {
            AfmVal z = 0;
            vector<AfmVal> p(K, 0);
            for (Node *x = X[i]; x != X[i+1]; x++)
            {
                z += w[x->idx]*x->val;
                AfmVal uu = 0;
                for (AfmIdx k = 0; k < K; k++)
                {
                    p[k] += U[x->idx*K+k]*x->val;
                    uu += U[x->idx*K+k]*U[x->idx*K+k];
                }
                z -= 0.5*uu*x->val*x->val;
            }

            for (AfmIdx k = 0; k < K; k++)
                z += 0.5*p[k]*p[k];

            const AfmVal yz = Y[i]*z;
            if (yz > 0)
            {
                const AfmVal exp_m = exp(-yz);
                sigma[i] = exp_m/(1+exp_m);
                F += log(1+exp_m);
            }
            else
            {
                const AfmVal exp_m = exp(yz);
                sigma[i] = 1/(1+exp_m);
                F += -yz+log(1+exp_m);
            }

            for (Node *x = X[i]; x != X[i+1]; x++)
            {
                g_w[x->idx] -= Y[i]*sigma[i]*x->val;
                for (AfmIdx k = 0; k < K; k++)
                {
                    const AfmIdx jk = x->idx*K+k;
                    AfmVal xi = p[k]-U[jk]*x->val;
                    g_U[jk] -= Y[i]*sigma[i]*xi*x->val;
                }
            }
        }

        if (lambda_w < 0)
            for (AfmIdx j = 0; j < J; j++)
                g_w[j] = 0;

        // 2-1. Stopping condition
        if (norm_g_0 < 0)
        {
            norm_g_0 = 0;
            for (AfmIdx j = 0; j < J; j++)
            {
                norm_g_0 += g_w[j]*g_w[j];
                for (AfmIdx k = 0; k < K; k++)
                    norm_g_0 += g_U[j*K+k]*g_U[j*K+k];
            }
            norm_g_0 = sqrt(norm_g_0);
        }

        norm_g = 0;
        for (AfmIdx j = 0; j < J; j++)
        {
            norm_g += g_w[j]*g_w[j];
            for (AfmIdx k = 0; k < K; k++)
                norm_g += g_U[j*K+k]*g_U[j*K+k];
        }
        norm_g = sqrt(norm_g);

        if (norm_g > tol*norm_g_0)
        {
            // 2-2. Calculate descent direction
            for (AfmIdx j = 0; j < J; j++)
            {
                s_w[j] = -g_w[j];
                for (AfmIdx k = 0; k < K; k++)
                {
                    const AfmIdx jk = j*K+k;
                    s_U[jk] = -g_U[jk];
                }
            }

            // 2-3. Prepare for line search
            AfmVal phi = 0;
            AfmVal F_theta = 0;
            AfmVal theta = 1/kappa;
            AfmIdx nr_searches1 = -1;

            for (AfmIdx j = 0; j < J; j++)
            {
                phi -= g_w[j]*s_w[j];
                for (AfmIdx k = 0; k < K; k++)
                {
                    const AfmIdx jk = j*K+k;
                    phi -= g_U[jk]*s_U[jk];
                }
            }
            phi *= nu;

            // 2-4. Do line search
            do
            {
                theta *= kappa;
                nr_searches1++;
                if (theta == 1)
                {
                    for (AfmIdx j = 0; j < J; j++)
                    {
                        w[j] += theta*s_w[j];
                        for (AfmIdx k = 0; k < K; k++)
                        {
                            const AfmIdx jk = j*K+k;
                            U[jk] += theta*s_U[jk];
                        }
                    }
                }
                else
                {
                    for (AfmIdx j = 0; j < J; j++)
                    {
                        w[j] -= (1-kappa)/kappa*theta*s_w[j];
                        for (AfmIdx k = 0; k < K; k++)
                        {
                            const AfmIdx jk = j*K+k;
                            U[jk] -= (1-kappa)/kappa*theta*s_U[jk];
                        }
                    }
                }

                F_theta = 0;
                for (AfmIdx j = 0; j < J; j++)
                {
                    F_theta += lambda_w*w[j]*w[j];
                    for (AfmIdx k = 0; k < K; k++)
                    {
                        const AfmIdx jk = j*K+k;
                        F_theta += lambda_U*U[jk]*U[jk];
                    }
                }
                F_theta *= 0.5;

                for (AfmIdx i = 0; i < I; i++)
                {
                    const AfmVal yz = Y[i]*fm_predict(X[i], X[i+1], model, true);
                    if (yz > 0)
                        F_theta += log(1+exp(-yz));
                    else
                        F_theta += -yz+log(1+exp(yz));
                }
            }
            while(F-F_theta < theta*phi);
            nr_searches = nr_searches1;
        }

        auto iter_time = omp_get_wtime()-start_time;
        total_time += iter_time;

        ///////////////////////////////////////////////////
        // Output information
        ///////////////////////////////////////////////////
        AfmVal norm_w = 0;
        AfmVal norm_U = 0;
        AfmVal loss = 0;

        for (AfmIdx j = 0; j < J; j++)
        {
            norm_w += w[j]*w[j];
            for (AfmIdx k = 0; k < K; k++)
                norm_U += U[j*K+k]*U[j*K+k];
        }
        for (AfmIdx i = 0; i < I; i++)
        {
            const AfmVal yz = Y[i]*fm_predict(X[i], X[i+1], model, true);
            if (yz > 0)
                loss += log(1.0+exp(-yz));
            else
                loss += -yz+log(1.0+exp(yz));
        }

        AfmVal loss_test = data_test.I <= 0? -1: evaluate(data_test, model, true);
        AfmVal obj = 0.5*(lambda_w*norm_w+lambda_U*norm_U)+loss;

        cout << "iter: " << std::setfill(' ') << std::setw(4) << t
             << " ls: " << std::setw(2) << nr_searches
             << " time: " << std::setw(9) << std::setprecision(8) << total_time
             << " obj: " << std::setw(9) << std::setprecision(8) << obj
             << " |g|: " << std::setw(9) << std::setprecision(8) << norm_g
             << " tr_loss: " << std::setw(9) << std::setprecision(8) << loss/data.I
             << " te_loss: " << std::setw(9) << std::setprecision(8) << loss_test
             << endl;
    }
}

void train_fm_als_gd(
    const AfmData &data,
    const AfmData &data_test,
    const Parameter param,
    const bool use_freq,
    AfmModel &model)
{
    if (data.J != model.J or data.I < 1)
        return;
    const AfmIdx T = param.iters;
    const AfmIdx I = data.I;
    const AfmIdx J = model.J;
    const AfmIdx K = model.K;
    const Node* const *X = data.indexes.data();
    const AfmVal *Y = data.labels.data();
    const AfmVal kappa = param.kappa;
    const AfmVal nu = param.nu;
    const AfmVal lambda_w = param.lambda_w;
    const AfmVal lambda_U = param.lambda_U;
    AfmVal *w = model.w.data();
    AfmVal *U = model.U.data();
    vector<AfmVal> freqs(J, 0);
    vector<AfmVal> z_bar(I, 0);
    vector<AfmVal> z_hat(I, 0);
    vector<AfmVal> z(I, 0);
    vector<AfmVal> sigma(I, 0);
    vector<AfmVal> P(I*K, 0);

    if (use_freq)
    {
        for (AfmIdx j = 0; j < J; j++)
            for(const Node *node = X[j]; node != X[j+1]; node++)
                freqs[j]++;
        for (AfmIdx j = 0; j < J; j++)
            if (freqs[j] == 0)
                freqs[j] = 1;
    }
    else
    {
        for (AfmIdx j = 0; j < J; j++)
            freqs[j] = 1;
    }

    for (AfmIdx j = 0; j < J; j++)
    {
        for (const Node *node = X[j]; node != X[j+1]; node++)
        {
            const AfmIdx i = node->idx;
            const AfmVal x = node->val;
            z_bar[i] += w[j]*x;
            for (AfmIdx k = 0; k < K; k++)
            {
                const AfmIdx jk = j*K+k;
                P[i*K+k] += U[jk]*x;
                z_hat[i] += U[jk]*U[jk]*x*x;
            }
        }
    }

    double total_time = 0;
    for (AfmIdx t = 0; t < T; t++)
    {
        auto start_time = omp_get_wtime();
        AfmIdx nr_searches_w = 0;
        AfmIdx nr_searches_U = 0;

        // update w
        if (lambda_w >= 0)
        {
            // compute objective & gradient
            AfmVal f_w = 0;
            vector<AfmVal> g_w(J, 0);
            for (AfmIdx j = 0; j < J; j++)
            {
                const AfmVal lambda_w_j = lambda_w*freqs[j];
                f_w += lambda_w_j*w[j]*w[j];
                g_w[j] += lambda_w_j*w[j];
            }
            f_w *= 0.5;
            for (AfmIdx i = 0; i < I; i++)
            {
                AfmVal pp = 0;
                for (AfmIdx k = 0; k < K; k++)
                {
                    const AfmIdx ik = i*K+k;
                    pp += P[ik]*P[ik];
                }
                z[i] = z_bar[i]+0.5*(pp-z_hat[i]);
                const AfmVal yz = Y[i]*z[i];
                if (yz > 0)
                {
                    const AfmVal exp_m = exp(-yz);
                    sigma[i] = exp_m/(1+exp_m);
                    f_w += log(1+exp_m);
                }
                else
                {
                    const AfmVal exp_m = exp(yz);
                    sigma[i] = 1/(1+exp_m);
                    f_w += -yz+log(1+exp_m);
                }
            }
            for (AfmIdx j = 0; j < J; j++)
                for (const Node *node = X[j]; node != X[j+1]; node++)
                    g_w[j] -= Y[node->idx]*sigma[node->idx]*node->val;

            // line search
            AfmVal phi_w = 0;
            AfmVal theta_w = 1/kappa;
            AfmVal f_w_theta = 1e300;
            vector<AfmVal> delta(I, 0);

            for (AfmIdx j = 0; j < J; j++)
                phi_w += g_w[j]*g_w[j];

            for (AfmIdx j = 0; j < J; j++)
                for (const Node *node = X[j]; node != X[j+1]; node++)
                    delta[node->idx] += g_w[j]*node->val;

            AfmIdx nr_searches = -1;
            while (f_w-f_w_theta < nu*theta_w*phi_w)
            {
                nr_searches++;
                theta_w *= kappa;
                f_w_theta = 0;
                for (AfmIdx j = 0; j < J; j++)
                {
                    const AfmVal w_j = w[j]-theta_w*g_w[j];
                    f_w_theta += lambda_w*freqs[j]*w_j*w_j;
                }
                f_w_theta *= 0.5;
                for (AfmIdx i = 0; i < I; i++)
                {
                    const AfmVal yz = Y[i]*(z[i]-theta_w*delta[i]);
                    if (yz > 0)
                        f_w_theta += log(1.0+exp(-yz));
                    else
                        f_w_theta += -yz+log(1.0+exp(yz));
                }
            }
            nr_searches_w += nr_searches;

            // update model and cached variables
            for (AfmIdx j = 0; j < J; j++)
                w[j] = w[j]-theta_w*g_w[j];
            for (AfmIdx i = 0; i < I; i++)
                z_bar[i] -= theta_w*delta[i];
        }

        if (lambda_U >= 0)
        {
            // update u_j
            for (AfmIdx j = 0; j < J; j++)
            {
                // compute objective & gradient
                const AfmVal lambda_U_j = lambda_U*freqs[j];
                AfmVal f_u = 0;
                AfmVal uu = 0;
                vector<AfmVal> g_u(K, 0);
                for (AfmIdx k = 0; k < K; k++)
                {
                    const AfmVal u = U[j*K+k];
                    uu += u*u;
                    g_u[k] = lambda_U_j*u;
                }
                f_u = 0.5*lambda_U_j*uu;
                for (const Node * node = X[j]; node != X[j+1]; node++)
                {
                    const AfmIdx i = node->idx;
                    const AfmVal x = node->val;
                    AfmVal pp = 0;
                    for (AfmIdx k = 0; k < K; k++)
                    {
                        const AfmVal p = P[i*K+k];
                        pp += p*p;
                    }
                    z[i] = z_bar[i]+0.5*(pp-z_hat[i]);
                    const AfmVal yz = Y[i]*z[i];
                    if (yz > 0)
                    {
                        const AfmVal exp_m = exp(-yz);
                        sigma[i] = exp_m/(1+exp_m);
                        f_u += log(1+exp_m);
                    }
                    else
                    {
                        const AfmVal exp_m = exp(yz);
                        sigma[i] = 1/(1+exp_m);
                        f_u += -yz+log(1+exp_m);
                    }
                    for (AfmIdx k = 0; k < K; k++)
                        g_u[k] -= Y[i]*sigma[i]*(P[i*K+k]-U[j*K+k]*x)*x;
                }

                AfmVal phi_u = 0;
                AfmVal theta_u = 1.0/kappa;
                AfmVal f_u_theta = 1e300;

                for (AfmIdx k = 0; k < K; k++)
                    phi_u += g_u[k]*g_u[k];

                if (phi_u == 0)
                    continue;

                AfmVal uu_prime = 0;
                AfmIdx nr_searches = -1;
                while(f_u-f_u_theta < nu*theta_u*phi_u)
                {
                    if (theta_u < 1e-20)
                    {
                        theta_u = 0;
                        break;
                    }
                    nr_searches++;
                    theta_u *= kappa;
                    uu_prime = 0;
                    for (AfmIdx k = 0; k < K; k++)
                    {
                        const AfmVal u_prime = U[j*K+k]-theta_u*g_u[k];
                        uu_prime += u_prime*u_prime;
                    }
                    f_u_theta = 0.5*lambda_U_j*uu_prime;
                    for (const Node * node = X[j]; node != X[j+1]; node++)
                    {
                        const AfmIdx i = node->idx;
                        const AfmVal x = node->val;
                        AfmVal pp_prime = 0;
                        for (AfmIdx k = 0; k < K; k++)
                        {
                            const AfmVal p_prime = P[i*K+k]-theta_u*g_u[k]*x;
                            pp_prime += p_prime*p_prime;
                        }
                        const AfmVal z_prime = z_bar[i]+0.5*(pp_prime-z_hat[i]-(uu_prime-uu)*x*x);
                        const AfmVal yz_prime = Y[i]*z_prime;
                        if (yz_prime > 0)
                            f_u_theta += log(1.0+exp(-yz_prime));
                        else
                            f_u_theta += -yz_prime+log(1.0+exp(yz_prime));
                    }
                }
                nr_searches_U += nr_searches;

                if (theta_u > 0)
                {
                    // update variables
                    for (AfmIdx k = 0; k < K; k++)
                        U[j*K+k] -= theta_u*g_u[k];

                    for (const Node * node = X[j]; node != X[j+1]; node++)
                    {
                        const AfmIdx i = node->idx;
                        const AfmVal x = node->val;
                        for (AfmIdx k = 0; k < K; k++)
                            P[i*K+k] -= theta_u*g_u[k]*x;
                        z_hat[i] += (uu_prime-uu)*x*x;
                    }
                }
            }
        }

        auto iter_time = omp_get_wtime()-start_time;
        total_time += iter_time;

        ///////////////////////////////////////////////////
        // Output information
        ///////////////////////////////////////////////////

        {
            AfmVal norm_w = 0;
            AfmVal norm_U = 0;
            AfmVal loss = 0;

            for (AfmIdx j = 0; j < J; j++)
            {
                if (lambda_w > 0)
                {
                    const AfmVal lambda_w_j = use_freq? lambda_w*freqs[j]: lambda_w;
                    norm_w += lambda_w_j*w[j]*w[j];
                }
                if (lambda_U > 0)
                {
                    const AfmVal lambda_U_j = use_freq? lambda_U*freqs[j]: lambda_U;
                    for (AfmIdx k = 0; k < K; k++)
                        norm_U += lambda_U_j*U[j*K+k]*U[j*K+k];
                }
            }
            for (AfmIdx i = 0; i < data.I; i++)
            {
                vector<AfmVal> p(K, 0);
                AfmVal z = 0;
                for (AfmIdx k = 0; k < K; k++)
                    z += P[i*K+k]*P[i*K+k];
                z = z_bar[i]+0.5*(z-z_hat[i]);
                loss += log(1.0+exp(-Y[i]*z));
            }

            AfmVal obj = 0.5*(norm_w+norm_U)+loss;
            AfmVal loss_test = data_test.I <= 0? -1: evaluate(data_test, model, true);
            cout << "iter: " << std::setfill(' ') << std::setw(4) << t
                 << " time: " << std::setw(9) << std::setprecision(8) << total_time
                 << " ls(w): " << std::setw(2) << nr_searches_w
                 << " ls(U): " << std::setw(2) << (double)nr_searches_U/(double)J
                 << " obj: " << std::setw(9) << std::setprecision(8) << obj
                 << " tr_loss: " << std::setw(9) << std::setprecision(8) << loss/data.I
                 << " te_loss: " << std::setw(9) << std::setprecision(8) << loss_test
                 << endl;
        }
    }
}
