/*
 * Copyright 2011 Mads R. B. Kristensen <madsbk@gmail.com>
 *
 * This file is part of DistNumPy <https://github.com/distnumpy>.
 *
 * DistNumPy is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DistNumPy is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DistNumPy. If not, see <http://www.gnu.org/licenses/>.
 */


/*===================================================================
 *
 * Returns True if there is data conflict/overlap.
 */
static char dndnode_conflict(const dndop *A, const int Aidx,
                             const dndop *B, const int Bidx)
{
    npy_intp d;
    char Aaccesstype = A->accesstypes[Aidx];
    dndarray *Abase = A->views[Aidx]->base;
    char Baccesstype = B->accesstypes[Bidx];
    dndarray *Bbase = B->views[Bidx]->base;
    dndsvb *Asvb = A->svbs[Aidx];
    dndsvb *Bsvb = B->svbs[Bidx];

    if(Abase->uid == Bbase->uid)
        if(Aaccesstype == DNPY_WRITE || Baccesstype == DNPY_WRITE)
        {
            if(Asvb == NULL || Bsvb == NULL)
                return 1;//Depend on the whole array.

            char conflict = 1;
            for(d=0; d<Abase->ndims; d++)
            {
                if(Bsvb->start[d] >=
                   Asvb->start[d] + Asvb->nsteps[d]
                   ||
                   Asvb->start[d] >=
                   Bsvb->start[d] + Bsvb->nsteps[d])
                {
                    conflict = 0;
                    break;//No conflict at the svb level.
                }
            }
            if(conflict)
                return 1;
        }
    return 0;
}/* dndnode_conflict */

/*===================================================================
 *
 * Add the list of node to the dependency system.
 * Node that all nodes in the list must relate to the same operation.
 */
void dep_add(dndnode *nodes, int nnodes, int force_laziness)
{
    npy_intp i;
    int n, idx;

    #ifdef DNDY_TIME
        unsigned long long tdelta;
        DNDTIME(tdelta);
    #endif

    //Init values for statustics.
    #ifdef DNPY_STATISTICS
        for(n=0; n<nnodes; n++)
            nodes[n].uid = ++node_uid_count;
        nodes[0].op->uid = ++op_uid_count;
    #endif
    //Handle one node at a time.
    for(n=0; n<nnodes; n++)
    {
        dndnode *node = &nodes[n];
        assert(node->op != NULL);
        assert(node->op_ary_idx >= 0);
        assert(node->op_ary_idx < node->op->narys);
        assert(nodes[0].op == node->op);
        node->next = NULL;
        idx = node->op_ary_idx;
        //We append the new node to the linked list and for each
        //conflict we increase the refcount of the new node.
        if(node->op->svbs[idx] == NULL)//Whole array.
        {
            dndarray *ary = node->op->views[idx]->base;
            for(i=0; i<ary->nblocks; i++)
            {
                if(ary->rootnodes[i] == NULL)
                    ary->rootnodes[i] = node;
                else
                {
                    dndnode *tnode = ary->rootnodes[i];
                    while(1)
                    {
                        if(tnode->op != node->op)
                            node->op->refcount++;
                        if(tnode->next == NULL)//We are finished.
                            break;
                        tnode = tnode->next;//Go to next node.
                    }
                    tnode->next = node;
                    assert(tnode->next->next == NULL);
                }
                //Need to clone the new node to get it "spanning" over
                //the whole array.
                memcpy(workbuf_nextfree, node, sizeof(dndnode));
                node = workbuf_nextfree;
                WORKBUF_INC(sizeof(dndnode));

                #ifdef DNPY_STATISTICS
                    node->uid = ++node_uid_count;
                #endif
            }
        }
        else
        {
            assert(node->op->svbs[idx]->rootnode != NULL);
            dndnode *tnode = *node->op->svbs[idx]->rootnode;
            if(tnode == NULL)
                *node->op->svbs[idx]->rootnode = node;
            else
            {
                while(1)
                {
                    if(tnode->op != node->op &&
                       dndnode_conflict(node->op, idx, tnode->op,
                                        tnode->op_ary_idx))
                        node->op->refcount++;

                    if(tnode->next == NULL)//We are finished.
                        break;
                    tnode = tnode->next;//Go to next node.
                }
                tnode->next = node;
            }
        }
    }

    //Place the operation in the ready queue when no dependency was
    //found.
    if(nodes[0].op->refcount == 0)
    {
        assert(ready_queue_size+1 <= DNPY_RDY_QUEUE_MAXSIZE);
        ready_queue[ready_queue_size++] = nodes[0].op;
    }

    assert(ready_queue_size > 0);

    #ifdef DNDY_TIME
        DNDTIME_SUM(tdelta, dndt.dag_svb_add)
    #endif

    #ifdef DNPY_NO_LAZY_EVAL
    if(!force_laziness)
        dag_svb_flush(0);//Note that we do not free the work buffer
    #endif
} /* dep_add */


/*===================================================================
 *
 * Removes a operation from the dependency system.
 * op2apply is the operations that should be applyed.
 * If op2apply is NULL then op2apply is ignored.
 * Returns number of operations in op2apply.
 */
npy_intp dep_remove(dndop *op, dndop *op2apply[])
{
    int j;
    npy_intp b, i, nops=1, mnops=DNPY_MAX_OP_MERGES;

    #ifdef DNDY_TIME
        unsigned long long tdelta;
        DNDTIME(tdelta);
    #endif

    if(op2apply != NULL)
        op2apply[0] = op;
    else
        mnops = 1;

    for(i=0; i<nops; i++)
    {
        if(op2apply != NULL)
            op = op2apply[i];

        for(j=0; j<op->narys; j++)
        {
            if(op->svbs[j] == NULL)//Whole array.
            {
                dndarray *ary = op->views[j]->base;
                assert(ary != NULL);
                for(b=0; b<ary->nblocks; b++)
                {
                    assert(ary->rootnodes[b] != NULL);
                    while(ary->rootnodes[b] != NULL &&
                          ary->rootnodes[b]->op == op)
                        ary->rootnodes[b] = ary->rootnodes[b]->next;

                    dndnode *n1 = ary->rootnodes[b];

                    //We are finished if the list has become empty.
                    if(n1 == NULL)
                        continue;

                    //Handle the first node in the list.
                    if(--n1->op->refcount == 0)
                    {
                        if(nops < mnops && n1->op->optype == DNPY_NONCOMM)
                            op2apply[nops++] = n1->op;
                        else
                            ready_queue[ready_queue_size++] = n1->op;
                    }
                    //Handle the rest of the nodes in the list.
                    dndnode *n2 = n1->next;
                    while(n2 != NULL)
                    {
                        assert(n1->next == n2);
                        if(n2->op == op)//Remove the node.
                            n1->next = n2->next;
                        else
                        {
                            if(--n2->op->refcount == 0)
                            {
                                if(nops < mnops && n2->op->optype == DNPY_NONCOMM)
                                    op2apply[nops++] = n2->op;
                                else
                                    ready_queue[ready_queue_size++] = n2->op;
                            }
                            n1 = n2;
                        }
                        n2 = n2->next;
                    }
                }
            }
            else
            {
                dndsvb *svb = op->svbs[j];
                while((*svb->rootnode) != NULL &&
                      (*svb->rootnode)->op == op)
                    *svb->rootnode = (*svb->rootnode)->next;
                dndnode *n1 = *svb->rootnode;

                //We are finished if the list has become empty.
                if(n1 == NULL)
                    continue;

                //Handle the first node in the list.
                if(dndnode_conflict(op, j, n1->op, n1->op_ary_idx))
                    if(--n1->op->refcount == 0)
                    {
                        if(nops < mnops && n1->op->optype == DNPY_NONCOMM)
                            op2apply[nops++] = n1->op;
                        else
                            ready_queue[ready_queue_size++] = n1->op;
                    }

                //Handle the rest of the nodes in the list.
                dndnode *n2 = n1->next;
                while(n2 != NULL)
                {
                    assert(n1->next == n2);
                    if(n2->op == op)//Remove the node.
                        n1->next = n2->next;
                    else
                    {
                        if(dndnode_conflict(op, j, n2->op, n2->op_ary_idx))
                            if(--n2->op->refcount == 0)
                            {
                                if(nops < mnops && n2->op->optype == DNPY_NONCOMM)
                                    op2apply[nops++] = n2->op;
                                else
                                    ready_queue[ready_queue_size++] = n2->op;
                            }
                        n1 = n2;
                    }
                    n2 = n2->next;//Go to next node.
                }
            }
        }
    }

    #ifdef DNDY_TIME
        DNDTIME_SUM(tdelta, dndt.dag_svb_rm)
    #endif

    return nops;
}/* dep_remove */


/*===================================================================
 *
 * Flush the dependency system.
 * Frees the work buffer when 'free_workbuf' is true.
 */
void dep_flush(int free_workbuf)
{
    npy_intp i, j, f, commsize, ncommsize;
    int fcomm[DNPY_RDY_QUEUE_MAXSIZE];
    int fcommsize;
    MPI_Request reqs[DNPY_RDY_QUEUE_MAXSIZE];
    MPI_Status reqstatus[DNPY_RDY_QUEUE_MAXSIZE];
    dndop *comm[DNPY_RDY_QUEUE_MAXSIZE];
    dndop *ncomm[DNPY_RDY_QUEUE_MAXSIZE];
    MPI_Datatype dtype[DNPY_RDY_QUEUE_MAXSIZE];
    npy_intp dtypesize=0;

    #ifdef DNDY_TIME
        ++dndt.nflush;
        unsigned long long tdelta;
        DNDTIME(tdelta);
    #endif
    #ifdef DNPY_STATISTICS
        dag_svb_dump();
    #endif

    commsize=0; ncommsize=0;
    while(ready_queue_size + commsize + ncommsize > 0)
    {
        #ifdef DNDY_TIME
            unsigned long long tdelta2;
            DNDTIME(tdelta2);
        #endif

        assert(ready_queue_size <= DNPY_RDY_QUEUE_MAXSIZE);
        //Sort the queue into two queues - one for communication and
        //one for non-communication nodes.
        //Furthermore, initiate the communication nodes.
        for(i=0; i<ready_queue_size; i++)
        {
            assert(ready_queue[i]->refcount == 0);
            //Init. all communication nodes in the ready queue.
            if(ready_queue[i]->optype == DNPY_COMM)
            {
                dndop_comm *C = (dndop_comm*) ready_queue[i];
                MPI_Datatype comm_dtype;
                assert(C->refcount == 0);
                switch(C->op)
                {
                    case DNPY_RECV:
                        assert(C->narys == 1);
                        comm_dtype = calc_svb_MPIdatatype(C->views[0],
                                                          C->svbs[0]);
                        delayed_array_allocation(C->views[0]->base);
                        MPI_Irecv(C->views[0]->base->data +
                                  C->svbs[0]->comm_offset,
                                  1, comm_dtype,
                                  C->remote_rank, C->mpi_tag,
                                  MPI_COMM_WORLD, &reqs[commsize]);
                        MPI_Type_free(&comm_dtype);
                        break;
                    case DNPY_BUF_RECV:
                        assert(C->narys == 1);
                        assert(C->svbs[0]->data == NULL);
                        C->svbs[0]->data = workbuf_nextfree;
                        WORKBUF_INC(C->svbs[0]->nelem *
                                    C->views[0]->base->elsize);
                        assert(C->svbs[0]->data != NULL);
                        MPI_Irecv(C->svbs[0]->data, C->svbs[0]->nelem,
                                  C->views[0]->base->mpi_dtype,
                                  C->remote_rank, C->mpi_tag,
                                  MPI_COMM_WORLD, &reqs[commsize]);
                        break;
                    case DNPY_SEND:
                        assert(C->narys == 1);
                        comm_dtype = calc_svb_MPIdatatype(C->views[0],
                                                          C->svbs[0]);
                        delayed_array_allocation(C->views[0]->base);
                        MPI_Isend(C->views[0]->base->data +
                                  C->svbs[0]->comm_offset,
                                  1, comm_dtype,
                                  C->remote_rank, C->mpi_tag,
                                  MPI_COMM_WORLD, &reqs[commsize]);
                        dtype[dtypesize++] = comm_dtype;
                        //At the moment we have to delay this freeing to
                        //the end of the flush. I’m not sure if this is
                        //a bug in DistNumPy or the MPICH-2 implementa-
                        //tion.
                        //MPI_Type_free(&comm_dtype);
                        break;
                    case DNPY_BUF_SEND:
                        assert(C->narys == 1);
                        assert(C->svbs[0]->data != NULL);
                        MPI_Isend(C->svbs[0]->data, C->svbs[0]->nelem,
                                  C->views[0]->base->mpi_dtype,
                                  C->remote_rank, C->mpi_tag,
                                  MPI_COMM_WORLD, &reqs[commsize]);
                        break;
                    case DNPY_REDUCE_SEND:
                        assert(C->narys == 1);
                        comm_dtype = calc_svb_MPIdatatype(C->views[0],
                                                          C->svbs[0]);
                        assert(C->svbs[0]->data != NULL);
                        MPI_Isend(C->svbs[0]->data +
                                  C->svbs[0]->comm_offset,
                                  1, comm_dtype,
                                  C->remote_rank, C->mpi_tag,
                                  MPI_COMM_WORLD, &reqs[commsize]);
                        dtype[dtypesize++] = comm_dtype;
                        //At the moment we have to delay this freeing to
                        //the end of the flush. I’m not sure if this is
                        //a bug in DistNumPy or the MPICH-2 implementa-
                        //tion.
                        //MPI_Type_free(&comm_dtype);
                        break;
                    case DNPY_COPY_INTO:
                        if(C->narys == 1)
                        {
                            if(C->accesstypes[0] == DNPY_WRITE)
                            {
                                comm_dtype = calc_svb_MPIdatatype(C->views[0],
                                                                  C->svbs[0]);
                                delayed_array_allocation(C->views[0]->base);
                                MPI_Irecv(C->views[0]->base->data +
                                          C->svbs[0]->comm_offset,
                                          1, comm_dtype,
                                          C->remote_rank, C->mpi_tag,
                                          MPI_COMM_WORLD, &reqs[commsize]);
                                MPI_Type_free(&comm_dtype);
                            }
                            else
                            {
                                assert(C->accesstypes[0] == DNPY_READ);
                                comm_dtype = calc_svb_MPIdatatype(C->views[0],
                                                                  C->svbs[0]);
                                delayed_array_allocation(C->views[0]->base);
                                MPI_Isend(C->views[0]->base->data +
                                          C->svbs[0]->comm_offset,
                                          1, comm_dtype,
                                          C->remote_rank, C->mpi_tag,
                                          MPI_COMM_WORLD, &reqs[commsize]);
                                dtype[dtypesize++] = comm_dtype;
                                //At the moment we have to delay this freeing to
                                //the end of the flush. I’m not sure if this is
                                //a bug in DistNumPy or the MPICH-2 implementa-
                                //tion.
                                //MPI_Type_free(&comm_dtype);
                            }
                        }
                        break;
                    default:
                        fprintf(stderr, "Unknown DAG operation: %s.\n",
                                optype2str(C->op));
                        MPI_Abort(MPI_COMM_WORLD, -1);
                }
                comm[commsize++] = ready_queue[i];
            }
            else
            {
                assert(ready_queue[i]->optype == DNPY_NONCOMM);
                ncomm[ncommsize++] = ready_queue[i];
            }
        }
        //The ready queue is now empty.
        ready_queue_size = 0;

        #ifdef DNDY_TIME
            DNDTIME_SUM(tdelta2, dndt.comm_init)
        #endif

        //Apply one non-communication node and move new non-depend
        //nodes to the ready queue.
        //Instead of moving new non-depended non-communication nodes
        //to the ready queue they are directly applied.
        if(ncommsize > 0)
        {
            dndop *op = ncomm[--ncommsize];//Using a FILO order.
            dndop **op2apply = workbuf_nextfree;
            npy_intp nops = dep_remove(op, op2apply);
            npy_intp cur_op;
            npy_intp mreserved = nops * DNPY_WORK_BUFFER_MEM_ALIGNMENT;
            WORKBUF_INC(mreserved);//Reserve memory.

            for(cur_op=0; cur_op < nops; cur_op++)
            {
                op = op2apply[cur_op];
                switch(op->op)
                {
                    case DNPY_UFUNC:
                        //apply_ufunc((dndop_ufunc*)op);
                        Py_DECREF(((dndop_ufunc*)op)->PyOp);
                        break;
                    case DNPY_DESTROY_ARRAY:
                        assert(op->narys == 1);
                        assert(op->views[0] != NULL);
                        rm_dndview(op->views[0]->uid);
                        break;
                    default:
                        fprintf(stderr, "Unknown DAG operation: %s.\n",
                                optype2str(op->op));
                        MPI_Abort(MPI_COMM_WORLD, -1);
                }
            }
            workbuf_nextfree -= mreserved;//Unreserve memory.

            ++dndt.napply;
            dndt.nconnect += commsize;
            dndt.nconnect_max = MAX(dndt.nconnect_max, commsize);
        }
        //Test for ready communication and possibly move new non-depend
        //nodes to the ready queue. Furthermore, if there is nothing
        //else to do (no operations that are ready) we wait.
        if(commsize > 0)
        {
            #ifdef DNDY_TIME
                unsigned long long tdelta2;
                DNDTIME(tdelta2);
            #endif

            if(ncommsize > 0)
            {
                MPI_Testsome(commsize, reqs, &fcommsize, fcomm,
                             reqstatus);
                assert(fcommsize != MPI_UNDEFINED);
            }
            else if(ready_queue_size == 0)
            {
                MPI_Waitsome(commsize, reqs, &fcommsize, fcomm,
                             reqstatus);
                assert(fcommsize != MPI_UNDEFINED);
            }
            else
                fcommsize = 0;

            #ifdef DNDY_TIME
                DNDTIME_SUM(tdelta2, dndt.ufunc_comm)
            #endif

            for(f=0; f<fcommsize; f++)
            {
                dndop *op = comm[fcomm[f]];
                dep_remove(op, NULL);
            }
            for(f=0; f<fcommsize; f++)
            {
                j=0;
                for(i=0; i<commsize; i++)
                    if(fcomm[f]-f != i)
                    {
                        comm[j] = comm[i];
                        reqs[j++] = reqs[i];
                    }
            }
            commsize -= fcommsize;
        }
    }
    //Do the delayed MPI type freeing.
    for(i=0; i<dtypesize; i++)
        MPI_Type_free(&dtype[i]);

    assert(ready_queue_size == 0);
    #ifdef DNDY_TIME
        DNDTIME_SUM(tdelta, dndt.dag_svb_flush)
    #endif
    if(free_workbuf)
    {
        workbuf_nextfree = workbuf;
        WORKBUF_INC(1);//Making sure that the memory is aligned.
    }
}/* dep_flush */



