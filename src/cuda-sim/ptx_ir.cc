// Copyright (c) 2009-2021, Tor M. Aamodt, Ali Bakhoda, Wilson W.L. Fung,
// George L. Yuan, Vijay Kandiah, Nikos Hardavellas,
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern 
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "ptx_ir.h"
#include "ptx_parser.h"
typedef void *yyscan_t;
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <list>
#include "assert.h"
#include "opcodes.h"
#include "ptx.tab.h"

#include "../../libcuda/gpgpu_context.h"
#include "cuda-sim.h"

#define STR_SIZE 1024

/*
传入参数 PC 值，获取该 PC 值对应的PTX指令，该指令为 ptx_instruction 类的对象。s_g_pc_to_insn 如下定义：
    std::vector<ptx_instruction *>
          s_g_pc_to_insn;  // a direct mapping from PC to instruction
是 PC 值 --> ptx_instruction 的映射。
*/
const ptx_instruction *gpgpu_context::pc_to_instruction(unsigned pc) {
  if (pc < s_g_pc_to_insn.size())
    return s_g_pc_to_insn[pc];
  else
    return NULL;
}

unsigned symbol::get_uid() {
  unsigned result = (gpgpu_ctx->symbol_sm_next_uid)++;
  return result;
}

void symbol::add_initializer(const std::list<operand_info> &init) {
  m_initializer = init;
}

void symbol::print_info(FILE *fp) const {
  fprintf(fp, "uid:%u, decl:%s, type:%p, ", m_uid, m_decl_location.c_str(),
          m_type);
  if (m_address_valid) fprintf(fp, "<address valid>, ");
  if (m_is_label) fprintf(fp, " is_label ");
  if (m_is_shared) fprintf(fp, " is_shared ");
  if (m_is_const) fprintf(fp, " is_const ");
  if (m_is_global) fprintf(fp, " is_global ");
  if (m_is_local) fprintf(fp, " is_local ");
  if (m_is_tex) fprintf(fp, " is_tex ");
  if (m_is_func_addr) fprintf(fp, " is_func_addr ");
  if (m_function) fprintf(fp, " %p ", m_function);
}

symbol_table::symbol_table() { assert(0); }

symbol_table::symbol_table(const char *scope_name, unsigned entry_point,
                           symbol_table *parent, gpgpu_context *ctx) {
  gpgpu_ctx = ctx;
  m_scope_name = std::string(scope_name);
  m_reg_allocator = 0;
  m_shared_next = 0;
  m_const_next = 0;
  m_global_next = 0x100;
  m_local_next = 0;
  m_tex_next = 0;

  // Jin: handle instruction group for cdp
  m_inst_group_id = 0;

  m_parent = parent;
  if (m_parent) {
    m_shared_next = m_parent->m_shared_next;
    m_global_next = m_parent->m_global_next;
  }
}

void symbol_table::set_name(const char *name) {
  m_scope_name = std::string(name);
}

const ptx_version &symbol_table::get_ptx_version() const {
  if (m_parent == NULL)
    return m_ptx_version;
  else
    return m_parent->get_ptx_version();
}

unsigned symbol_table::get_sm_target() const {
  if (m_parent == NULL)
    return m_ptx_version.target();
  else
    return m_parent->get_sm_target();
}

void symbol_table::set_ptx_version(float ver, unsigned ext) {
  m_ptx_version = ptx_version(ver, ext);
}

void symbol_table::set_sm_target(const char *target, const char *ext,
                                 const char *ext2) {
  m_ptx_version.set_target(target, ext, ext2);
}

symbol *symbol_table::lookup(const char *identifier) {
  std::string key(identifier);
  std::map<std::string, symbol *>::iterator i = m_symbols.find(key);
  if (i != m_symbols.end()) {
    return i->second;
  }
  if (m_parent) {
    return m_parent->lookup(identifier);
  }
  return NULL;
}

symbol *symbol_table::add_variable(const char *identifier,
                                   const type_info *type, unsigned size,
                                   const char *filename, unsigned line) {
  char buf[1024];
  std::string key(identifier);
  assert(m_symbols.find(key) == m_symbols.end());
  snprintf(buf, 1024, "%s:%u", filename, line);
  symbol *s = new symbol(identifier, type, buf, size, gpgpu_ctx);
  m_symbols[key] = s;

  if (type != NULL && type->get_key().is_global()) {
    m_globals.push_back(s);
  }
  if (type != NULL && type->get_key().is_const()) {
    m_consts.push_back(s);
  }

  return s;
}

void symbol_table::add_function(function_info *func, const char *filename,
                                unsigned linenumber) {
  std::map<std::string, symbol *>::iterator i =
      m_symbols.find(func->get_name());
  if (i != m_symbols.end()) return;
  char buf[1024];
  snprintf(buf, 1024, "%s:%u", filename, linenumber);
  type_info *type = add_type(func);
  symbol *s = new symbol(func->get_name().c_str(), type, buf, 0, gpgpu_ctx);
  s->set_function(func);
  m_symbols[func->get_name()] = s;
}

// Jin: handle instruction group for cdp
symbol_table *symbol_table::start_inst_group() {
  char inst_group_name[4096];
  snprintf(inst_group_name, 4096, "%s_inst_group_%u", m_scope_name.c_str(),
           m_inst_group_id);

  // previous added
  assert(m_inst_group_symtab.find(std::string(inst_group_name)) ==
         m_inst_group_symtab.end());
  symbol_table *sym_table =
      new symbol_table(inst_group_name, 3 /*inst group*/, this, gpgpu_ctx);

  sym_table->m_global_next = m_global_next;
  sym_table->m_shared_next = m_shared_next;
  sym_table->m_local_next = m_local_next;
  sym_table->m_reg_allocator = m_reg_allocator;
  sym_table->m_tex_next = m_tex_next;
  sym_table->m_const_next = m_const_next;

  m_inst_group_symtab[std::string(inst_group_name)] = sym_table;

  return sym_table;
}

symbol_table *symbol_table::end_inst_group() {
  symbol_table *sym_table = m_parent;

  sym_table->m_global_next = m_global_next;
  sym_table->m_shared_next = m_shared_next;
  sym_table->m_local_next = m_local_next;
  sym_table->m_reg_allocator = m_reg_allocator;
  sym_table->m_tex_next = m_tex_next;
  sym_table->m_const_next = m_const_next;
  sym_table->m_inst_group_id++;

  return sym_table;
}

void register_ptx_function(const char *name,
                           function_info *impl);  // either libcuda or libopencl

bool symbol_table::add_function_decl(const char *name, int entry_point,
                                     function_info **func_info,
                                     symbol_table **sym_table) {
  std::string key = std::string(name);
  bool prior_decl = false;
  if (m_function_info_lookup.find(key) != m_function_info_lookup.end()) {
    *func_info = m_function_info_lookup[key];
    prior_decl = true;
  } else {
    *func_info = new function_info(entry_point, gpgpu_ctx);
    (*func_info)->set_name(name);
    (*func_info)->set_maxnt_id(0);
    m_function_info_lookup[key] = *func_info;
  }

  if (m_function_symtab_lookup.find(key) != m_function_symtab_lookup.end()) {
    assert(prior_decl);
    *sym_table = m_function_symtab_lookup[key];
  } else {
    assert(!prior_decl);
    *sym_table = new symbol_table("", entry_point, this, gpgpu_ctx);

    // Initial setup code to support a register represented as "_".
    // This register is used when an instruction operand is
    // not read or written.  However, the parser must recognize it
    // as a legitimate register but we do not want to pass
    // it to the micro-architectural register to the performance simulator.
    // For this purpose we add a symbol to the symbol table but
    // mark it as a non_arch_reg so it does not effect the performance sim.
    //用于支持表示为"_"的寄存器的初始设置代码。当未读取或写入指令操作数时，使用此寄存器。然而，解析器
    //必须将其识别为合法的寄存器，但我们不想将其传递给微体系结构寄存器，传递给性能模拟器。为此，我们向
    //符号表中添加一个符号，但将其标记为non_arch_reg，这样不会影响性能模拟。 
    //将带有"_"寄存器的指令中的"_"设置为null_key，并设置null_key.set_is_non_arch_reg()。
    type_info_key null_key(reg_space, 0, 0, 0, 0, 0);
    null_key.set_is_non_arch_reg();
    // First param is null - which is bad.
    // However, the first parameter is actually unread in the constructor...
    //第一个参数为空-这是错误的。然而，第一个参数实际上在构造函数中没有读。

    // TODO - remove the symbol_table* from type_info
    type_info *null_type_info = new type_info(NULL, null_key);
    symbol *null_reg =
        (*sym_table)->add_variable("_", null_type_info, 0, "", 0);
    null_reg->set_regno(0, 0);

    (*sym_table)->set_name(name);
    (*func_info)->set_symtab(*sym_table);
    m_function_symtab_lookup[key] = *sym_table;
    assert((*func_info)->get_symtab() == *sym_table);
    register_ptx_function(name, *func_info);
  }
  return prior_decl;
}

function_info *symbol_table::lookup_function(std::string name) {
  std::string key = std::string(name);
  std::map<std::string, function_info *>::iterator it =
      m_function_info_lookup.find(key);
  assert(it != m_function_info_lookup.end());
  return it->second;
}

type_info *symbol_table::add_type(memory_space_t space_spec,
                                  int scalar_type_spec, int vector_spec,
                                  int alignment_spec, int extern_spec) {
  if (space_spec == param_space_unclassified) space_spec = param_space_local;
  type_info_key t(space_spec, scalar_type_spec, vector_spec, alignment_spec,
                  extern_spec, 0);
  type_info *pt;
  pt = new type_info(this, t);
  return pt;
}

type_info *symbol_table::add_type(function_info *func) {
  type_info_key t;
  type_info *pt;
  t.set_is_func();
  pt = new type_info(this, t);
  return pt;
}

type_info *symbol_table::get_array_type(type_info *base_type,
                                        unsigned array_dim) {
  type_info_key t = base_type->get_key();
  t.set_array_dim(array_dim);
  type_info *pt = new type_info(this, t);
  // Where else is m_types being used? As of now, I dont find any use of it and
  // causing seg fault. So disabling m_types.
  // TODO: find where m_types can be used in future and solve the seg fault.
  // pt = m_types[t] = new type_info(this,t);
  return pt;
}

void symbol_table::set_label_address(const symbol *label, unsigned addr) {
  std::map<std::string, symbol *>::iterator i = m_symbols.find(label->name());
  assert(i != m_symbols.end());
  symbol *s = i->second;
  s->set_label_address(addr);
}

void symbol_table::dump() {
  printf("\n\n");
  printf("Symbol table for \"%s\":\n", m_scope_name.c_str());
  std::map<std::string, symbol *>::iterator i;
  for (i = m_symbols.begin(); i != m_symbols.end(); i++) {
    printf("%30s : ", i->first.c_str());
    if (i->second)
      i->second->print_info(stdout);
    else
      printf(" <no symbol object> ");
    printf("\n");
  }
  printf("\n");
}

unsigned operand_info::get_uid() {
  unsigned result = (gpgpu_ctx->operand_info_sm_next_uid)++;
  return result;
}

/*
find_next_real_instruction 用于找到下一条非is_label()的指令。如果i指向的指令是label，就再i++。
*/
std::list<ptx_instruction *>::iterator
function_info::find_next_real_instruction(
    std::list<ptx_instruction *>::iterator i) {
  while ((i != m_instructions.end()) && (*i)->is_label()) i++;
  return i;
}

/*
将各个指令分组为基本块（basic_block_t）。m_instructions 保存了 function_info 对象的所有PTX指
令。该函数执行完毕后，会将所有 PTX 指令分为基本块，添加到 m_basic_blocks 中。m_basic_blocks是
基本块类型的向量：std::vector<basic_block_t *> m_basic_blocks。例如下列 PTX 指令：
m_instructions保存了下列所有条指令：
    ld.param.u64 %rd18, [_Z6MatMulPiS_S_i_param_0];   |          -->ptx_begin/leaders[0]
    ......                                            |->m_basic_blocks[0]
    @%p1 bra $L__BB0_7;                               |          -->ptx_end

    add.s32 %r15, %r12, -1;                           |          -->ptx_begin/leaders[1]
    ......                                            |->m_basic_blocks[1]
    @%p2 bra $L__BB0_4;                               |          -->ptx_end

    sub.s32 %r33, %r12, %r35;                         |          -->ptx_begin/leaders[2]
    ......                                            |->m_basic_blocks[2]
    mul.wide.s32 %rd5, %r12, 4;                       |          -->ptx_end

    $L__BB0_3:                                        |          -->ptx_begin/leaders[3]
    ld.global.u32 %r18, [%rd30+-8];                   |->m_basic_blocks[3]
    ......                                            |
    @%p3 bra $L__BB0_3;                               |          -->ptx_end

    $L__BB0_4:                                        |          -->ptx_begin/leaders[4]
    setp.eq.s32 %p4, %r35, 0;                         |->m_basic_blocks[4]
    @%p4 bra $L__BB0_7;                               |          -->ptx_end

    mad.lo.s32 %r26, %r34, %r12, %r1;                 |          -->ptx_begin/leaders[5]
    ......                                            |->m_basic_blocks[5]
    add.s64 %rd32, %rd2, %rd26;                       |          -->ptx_end

    $L__BB0_6:                                        |          -->ptx_begin/leaders[6]
    .pragma "nounroll";                               |->m_basic_blocks[6]
    ......                                            |
    @%p5 bra $L__BB0_6;                               |          -->ptx_end

    $L__BB0_7:                                        |          -->ptx_begin/leaders[7]
    cvta.to.global.u64 %rd27, %rd17;                  |->m_basic_blocks[7]
    ......                                            |
    ret;                                              |          -->ptx_end
*/
void function_info::create_basic_blocks() {
  //leaders保存了一个代码块的首条指令。需要注意的是，一个代码块的结尾指令一般是 bra、ret、exit、
  //retp、break、call、callp 等指令，它们是跳转功能指令，代表了一个代码块的结尾；因此这条指令的
  //后面一条指令肯定是个新代码块的首条指令。
  std::list<ptx_instruction *> leaders;
  std::list<ptx_instruction *>::iterator i, l;

  // first instruction is a leader
  //m_instructions 的首条指令肯定属于第一个代码块，因此该首条指令是一个 leader。
  i = m_instructions.begin();
  leaders.push_back(*i);
  i++;
  //对m_instructions中除去首条指令之外的所有其余指令循环。
  while (i != m_instructions.end()) {
    //pI指向的是当前处理的指令。
    ptx_instruction *pI = *i;
    //is_label() 用于判断指令pI是否含有标签。label即为例如PTX指令块中的$L__BB0_6等：
    //  01.$L__BB0_6: <---- label
    //  02.  .pragma "nounroll";
    //  03.  ld.global.u32 %r28, [%rd32];
    //  04.  ...
    //  ...  ...
    //  12.  @%p5 bra $L__BB0_6; <---- label = $L__BB0_6
    if (pI->is_label()) {
      //如果pI是标签，代表它是代码块的首条指令，则直接将它压入leaders末端。
      leaders.push_back(pI);
      //find_next_real_instruction 用于找到下一条非is_label()的指令。
      i = find_next_real_instruction(++i);
    } else {
      //如果pI不是标签，需要判断操作码。因为，bra、ret、exit、retp、break、call、callp 这些指
      //令一般是一个代码块的结尾。
      switch (pI->get_opcode()) {
        //bra、ret、exit、retp、break一般是一个代码块的结尾，因此把 i++ 后的下一条指令直接压入
        //leaders末端。
        case BRA_OP:   //bra指令。
        case RET_OP:   //ret指令。
        case EXIT_OP:  //exit指令。
        case RETP_OP:  //retp指令。
        case BREAK_OP: //break指令。
          i++;
          if (i != m_instructions.end()) leaders.push_back(*i);
          i = find_next_real_instruction(i);
          break;
        case CALL_OP:  //call指令。
        case CALLP_OP: //callp指令。
          //如果该条指令有谓词寄存器，则一般是一个代码块的结尾，因此把 i++ 后的下一条指令直接压入
          //leaders末端。
          if (pI->has_pred()) {
            printf("GPGPU-Sim PTX: Warning found predicated call\n");
            i++;
            if (i != m_instructions.end()) leaders.push_back(*i);
            i = find_next_real_instruction(i);
          } else
            i++;
          break;
        default:
          i++;
      }
    }
  }

  //如果leaders为空，则该函数没有基本块。
  if (leaders.empty()) {
    printf("GPGPU-Sim PTX: Function \'%s\' has no basic blocks\n",
           m_name.c_str());
    return;
  }

  //bb_id是basic block的唯一标识，每次添加一个基本块的时候要加1。
  unsigned bb_id = 0;
  l = leaders.begin();
  i = m_instructions.begin();
  //m_basic_blocks是基本块类型的向量：
  //    std::vector<basic_block_t *> m_basic_blocks;
  //l指向的是由首条指令起始的第一个入口基本块，将该基本块加入到m_basic_blocks。
  //basic_block_t类在ptx_ir.h中定义，其构造函数：
  //  basic_block_t(unsigned ID, ptx_instruction *begin, ptx_instruction *end,
  //                bool entry, bool ex) {
  //    //basic block的唯一标识。
  //    bb_id = ID;
  //    //ptx_begin是该基本块的首条PTX指令。
  //    ptx_begin = begin;
  //    //ptx_end是该基本块的末尾PTX指令。
  //    ptx_end = end;
  //    //is_entry标志该基本块是否是入口处的基本块。
  //    is_entry = entry;
  //    //is_exit标志该基本块是否是出口处的基本块。
  //    is_exit = ex;
  //    immediatepostdominator_id = -1;
  //    immediatedominator_id = -1;
  //  }
  //因此 begin 指向的是 m_instructions 的第一条指令；*end 后面找到了再赋值；entry 置1；ex置0。
  //*find_next_real_instruction(i)是因为，前面 i 指向的是 m_instructions.begin()，而在之前把
  //基本块的首条指令加入到 m_basic_blocks 时，find_next_real_instruction(i) 是找到非label指令，
  //但我觉得这里没什么太大必要，因为一个 function_info 中的首条指令肯定不会是 label。
  m_basic_blocks.push_back(
      new basic_block_t(bb_id++, *find_next_real_instruction(i), NULL, 1, 0));
  //last_real_inst是处理leaders中的某条指令时，上一个指令由last_real_inst指向，代表上一个代码块
  //的结尾。l++是因为上面添加一个基本块时，l已经是一个基本块的开始指令，然后last_real_inst是用来
  //在处理下一个基本块的首条指令时，为上一个基本块的末尾指令ptx_end赋值。l++执行完毕后即指向了下一
  //个代码块的起始指令。
  ptx_instruction *last_real_inst = *(l++);
  //后面对 m_instructions 中的指令进行循环，依次处理每条指令所在的代码块。
  for (; i != m_instructions.end(); i++) {
    ptx_instruction *pI = *i;
    //如果 i 所指向的指令 == l 所指向的指令，即发现下一个基本块。因为上面l++执行完毕后即指向了下一
    //个代码块的起始指令。
    if (l != leaders.end() && *i == *l) {
      // found start of next basic block
      //为上一个基本块的 ptx_end 赋值为 last_real_inst。
      m_basic_blocks.back()->ptx_end = last_real_inst;
      if (find_next_real_instruction(i) !=
          m_instructions.end()) {  // if not bogus trailing label
        m_basic_blocks.push_back(new basic_block_t(
            bb_id++, *find_next_real_instruction(i), NULL, 0, 0));
        last_real_inst = *find_next_real_instruction(i);
      }
      // start search for next leader
      l++;
    }
    pI->assign_bb(m_basic_blocks.back());
    //对每一条指令循环时，循环一次，last_real_inst就赋值为当前指令 PI。这样在发现下一个代码块时，
    //就可以直接将上一个代码块的ptx_end赋值为last_real_inst。
    if (!pI->is_label()) last_real_inst = pI;
  }
  m_basic_blocks.back()->ptx_end = last_real_inst;
  m_basic_blocks.push_back(
      /*exit basic block*/ new basic_block_t(bb_id, NULL, NULL, 0, 1));
}

void function_info::print_basic_blocks() {
  printf("Printing basic blocks for function \'%s\':\n", m_name.c_str());
  std::list<ptx_instruction *>::iterator ptx_itr;
  unsigned last_bb = 0;
  for (ptx_itr = m_instructions.begin(); ptx_itr != m_instructions.end();
       ptx_itr++) {
    if ((*ptx_itr)->get_bb()) {
      if ((*ptx_itr)->get_bb()->bb_id != last_bb) {
        printf("\n");
        last_bb = (*ptx_itr)->get_bb()->bb_id;
      }
      printf("bb_%02u\t: ", (*ptx_itr)->get_bb()->bb_id);
      (*ptx_itr)->print_insn();
      printf("\n");
    }
  }
  printf("\nSummary of basic blocks for \'%s\':\n", m_name.c_str());
  std::vector<basic_block_t *>::iterator bb_itr;
  for (bb_itr = m_basic_blocks.begin(); bb_itr != m_basic_blocks.end();
       bb_itr++) {
    printf("bb_%02u\t:", (*bb_itr)->bb_id);
    if ((*bb_itr)->ptx_begin)
      printf(" first: %s\t", ((*bb_itr)->ptx_begin)->get_opcode_cstr());
    else
      printf(" first: NULL\t");
    if ((*bb_itr)->ptx_end) {
      printf(" last: %s\t", ((*bb_itr)->ptx_end)->get_opcode_cstr());
    } else
      printf(" last: NULL\t");
    printf("\n");
  }
  printf("\n");
}

void function_info::print_basic_block_links() {
  printf("Printing basic blocks links for function \'%s\':\n", m_name.c_str());
  std::vector<basic_block_t *>::iterator bb_itr;
  for (bb_itr = m_basic_blocks.begin(); bb_itr != m_basic_blocks.end();
       bb_itr++) {
    printf("ID: %d\t:", (*bb_itr)->bb_id);
    if (!(*bb_itr)->predecessor_ids.empty()) {
      printf("Predecessors:");
      std::set<int>::iterator p;
      for (p = (*bb_itr)->predecessor_ids.begin();
           p != (*bb_itr)->predecessor_ids.end(); p++) {
        printf(" %d", *p);
      }
      printf("\t");
    }
    if (!(*bb_itr)->successor_ids.empty()) {
      printf("Successors:");
      std::set<int>::iterator s;
      for (s = (*bb_itr)->successor_ids.begin();
           s != (*bb_itr)->successor_ids.end(); s++) {
        printf(" %d", *s);
      }
    }
    printf("\n");
  }
}

/*
找到 break 指令跳转到的目标基本块。
*/
operand_info *function_info::find_break_target(
    ptx_instruction *p_break_insn)  // find the target of a break instruction
{
  //break_bb指向的是 break 指令所在的基本块。
  const basic_block_t *break_bb = p_break_insn->get_bb();
  // go through the dominator tree
  //遍历必经结点树。
  for (const basic_block_t *p_bb = break_bb; p_bb->immediatedominator_id != -1;
       p_bb = m_basic_blocks[p_bb->immediatedominator_id]) {
    // reverse search through instructions in basic block for breakaddr
    // instruction
    unsigned insn_addr = p_bb->ptx_end->get_m_instr_mem_index();
    while (insn_addr >= p_bb->ptx_begin->get_m_instr_mem_index()) {
      ptx_instruction *pI = m_instr_mem[insn_addr];
      insn_addr -= 1;
      if (pI == NULL)
        continue;  // temporary solution for variable size instructions
      if (pI->get_opcode() == BREAKADDR_OP) {
        return &(pI->dst());
      }
    }
  }

  assert(0);

  // lazy fallback: just traverse backwards?
  for (int insn_addr = p_break_insn->get_m_instr_mem_index(); insn_addr >= 0;
       insn_addr--) {
    ptx_instruction *pI = m_instr_mem[insn_addr];
    if (pI->get_opcode() == BREAKADDR_OP) {
      return &(pI->dst());
    }
  }

  return NULL;
}

/*
将基本块连接起来，形成控制流图。
*/
void function_info::connect_basic_blocks()  // iterate across m_basic_blocks of
                                            // function, connecting basic blocks
                                            // together
{
  //代码基本块（basic block）迭代器。
  std::vector<basic_block_t *>::iterator bb_itr;
  std::vector<basic_block_t *>::iterator bb_target_itr;
  //m_basic_blocks的出口基本块。
  basic_block_t *exit_bb = m_basic_blocks.back();

  // start from first basic block, which we know is the entry point
  //从第一个基本块开始，我们知道这是切入点。
  bb_itr = m_basic_blocks.begin();
  //对每个代码基本块循环。
  for (bb_itr = m_basic_blocks.begin(); bb_itr != m_basic_blocks.end();
       bb_itr++) {
    //pI 指向的是当前基本块的 ptx_end 末尾指令。
    ptx_instruction *pI = (*bb_itr)->ptx_end;
    //(*bb_itr)->is_exit为真标志着，当前基本块是整个 function_info 的最后一个基本块，即出口点。
    //它的 ptx_end 不需要连接。
    if ((*bb_itr)->is_exit)  // reached last basic block, no successors to link
      continue;
    //ret指令：
    //    将执行返回到调用方的环境。发散返回将挂起线程，直到所有线程都准备好返回调用方。这允许多
    //    个不同的ret指令。常用方法：
    //           ret;
    //        @p ret;
    //exit指令：
    //    结束线程的执行。当线程退出时，系统将检查等待所有线程的障碍，以查看退出的线程是否是尚未
    //    到达barrier{.cta}（CTA中的所有线程）或barrier.cluster（群集中的所有线程）的唯一线程。
    //    如果退出线程阻挡了屏障，则释放屏障。常用方法：
    //           exit;
    //        @p exit;
    if (pI->get_opcode() == RETP_OP || pI->get_opcode() == RET_OP ||
        pI->get_opcode() == EXIT_OP) {
      //当前基本块 bb_itr 的后继者是 exit_bb。
      (*bb_itr)->successor_ids.insert(exit_bb->bb_id);
      //exit_bb 的前继者是 当前基本块 bb_itr。
      exit_bb->predecessor_ids.insert((*bb_itr)->bb_id);
      //如果 retp、ret、exit 指令有谓词寄存器，说明还有接下来一个基本块。
      if (pI->has_pred()) {
        printf("GPGPU-Sim PTX: Warning detected predicated return/exit.\n");
        // if predicated, add link to next block
        //通过 pI 在指令存储中的索引找下一条指令所在的基本块，连接过程同上。
        unsigned next_addr = pI->get_m_instr_mem_index() + pI->inst_size();
        if (next_addr < m_instr_mem_size && m_instr_mem[next_addr]) {
          basic_block_t *next_bb = m_instr_mem[next_addr]->get_bb();
          (*bb_itr)->successor_ids.insert(next_bb->bb_id);
          next_bb->predecessor_ids.insert((*bb_itr)->bb_id);
        }
      }
      continue;
    } else if (pI->get_opcode() == BRA_OP) {
      //带谓词寄存器的 bra 指令的连接。
      // find successor and link that basic_block to this one
      operand_info &target = pI->dst();  // get operand, e.g. target name
      unsigned addr = labels[target.name()];
      ptx_instruction *target_pI = m_instr_mem[addr];
      basic_block_t *target_bb = target_pI->get_bb();
      (*bb_itr)->successor_ids.insert(target_bb->bb_id);
      target_bb->predecessor_ids.insert((*bb_itr)->bb_id);
    }

    if (!(pI->get_opcode() == BRA_OP && (!pI->has_pred()))) {
      // if basic block does not end in an unpredicated branch,
      // then next basic block is also successor
      // (this is better than testing for .uni)
      //带谓词寄存器的 bra 指令的连接，非预测跳转。
      unsigned next_addr = pI->get_m_instr_mem_index() + pI->inst_size();
      basic_block_t *next_bb = m_instr_mem[next_addr]->get_bb();
      (*bb_itr)->successor_ids.insert(next_bb->bb_id);
      next_bb->predecessor_ids.insert((*bb_itr)->bb_id);
    } else
      assert(pI->get_opcode() == BRA_OP);
  }
}

/*
在该函数执行之前，已经执行过[基本块连接任务-connect_basic_blocks()]，但是这时候仅仅是按照每个基本块的
前后顺序进行了连接，没有针对 break 指令的跳转目标进行连接。因此下面的函数是分析PTX代码中的所有 break 
指令，并依据他们的跳转目标、或者是否是预测跳转来进行基本块连接上的修改。
*/
bool function_info::connect_break_targets()  // connecting break instructions
                                             // with proper targets
{
  //基本块迭代器。
  std::vector<basic_block_t *>::iterator bb_itr;
  std::vector<basic_block_t *>::iterator bb_target_itr;
  bool modified = false;

  // start from first basic block, which we know is the entry point
  //从第一个基本块开始，第一个基本块是函数的入口。
  bb_itr = m_basic_blocks.begin();
  for (bb_itr = m_basic_blocks.begin(); bb_itr != m_basic_blocks.end();
       bb_itr++) {
    basic_block_t *p_bb = *bb_itr;
    //pI指向的是基本块的最末尾指令。
    ptx_instruction *pI = p_bb->ptx_end;
    //如果p_bb指向的是最后一个基本块，即函数的出口的话，就没有后继结点需要连接到它。
    if (p_bb->is_exit)  // reached last basic block, no successors to link
      continue;
    //对操作码为 break 的指令进行处理。
    if (pI->get_opcode() == BREAK_OP) {
      // backup existing successor_ids for stability check
      //备份现有的successor_id以进行稳定性检查，orig_successor_ids指向的是现有的successor_id，即后
      //继结点的编号。
      std::set<int> orig_successor_ids = p_bb->successor_ids;

      // erase the previous linkage with old successors
      //删除之前执行[基本块连接任务-connect_basic_blocks()]时，p_bb与后继结点的链接。
      for (std::set<int>::iterator succ_ids = p_bb->successor_ids.begin();
           succ_ids != p_bb->successor_ids.end(); ++succ_ids) {
        basic_block_t *successor_bb = m_basic_blocks[*succ_ids];
        successor_bb->predecessor_ids.erase(p_bb->bb_id);
      }
      p_bb->successor_ids.clear();

      // find successor and link that basic_block to this one
      // successor of a break is set by an preceeding breakaddr instruction
      //找到后继结点，并将该 basic_block 链接到 p_bb。
      operand_info *target = find_break_target(pI);
      unsigned addr = labels[target->name()];
      ptx_instruction *target_pI = m_instr_mem[addr];
      basic_block_t *target_bb = target_pI->get_bb();
      p_bb->successor_ids.insert(target_bb->bb_id);
      target_bb->predecessor_ids.insert(p_bb->bb_id);

      //如果pI指向的PTX指令有谓词，则属于预测跳转。一方面，前面已经将其所在基本块与跳转到的目标基本块
      //进行连接，这是在预测成功的条件下；另一方面，如果预测失败，则仍旧沿着顺序基本块执行，因此还需要
      //将其与顺序下一个基本块进行连接。
      if (pI->has_pred()) {
        // predicated break - add link to next basic block
        unsigned next_addr = pI->get_m_instr_mem_index() + pI->inst_size();
        basic_block_t *next_bb = m_instr_mem[next_addr]->get_bb();
        p_bb->successor_ids.insert(next_bb->bb_id);
        next_bb->predecessor_ids.insert(p_bb->bb_id);
      }

      //返回是否由于 break 指令的存在，对基本块之间的连接进行了修改。
      modified = modified || (orig_successor_ids != p_bb->successor_ids);
    }
  }

  return modified;
}

/*
执行PDOM（分支处理中的后必经结点）检测。
*/
void function_info::do_pdom() {
  //将各个指令分组为基本块（basic_block_t）。
  create_basic_blocks();
  //将基本块连接起来，形成控制流图。
  connect_basic_blocks();
  bool modified = false;
  do {
    //寻找一个函数的完整PTX指令中，每一个（基本块）结点的必经结点（dominators）。
    find_dominators();
    //寻找一个函数的完整PTX指令中，每一个（基本块）结点的直接必经结点（immediate dominators）。
    find_idominators();
    //在该函数执行之前，已经执行过[基本块连接任务-connect_basic_blocks()]，但是这时候仅仅是按照每个
    //基本块的前后顺序进行了连接，没有针对 break 指令的跳转目标进行连接。因此下面的函数是分析PTX代码
    //中的所有 break 指令，并依据他们的跳转目标、或者是否是预测跳转来进行基本块连接上的修改。modified 
    //返回的是：是否由于 break 指令的存在，对基本块之间的连接进行了修改。
    modified = connect_break_targets();
  } while (modified == true);

  if (g_debug_execution >= 50) {
    print_basic_blocks();
    print_basic_block_links();
    print_basic_block_dot();
  }
  if (g_debug_execution >= 2) {
    print_dominators();
  }
  //寻找一个函数的完整PTX指令中，每一个（基本块）结点的后必经结点（post-dominators）。
  find_postdominators();
  //寻找一个函数的完整PTX指令中，每一个（基本块）结点的直接后必经结点（immediate post-dominators）。
  find_ipostdominators();
  if (g_debug_execution >= 50) {
    print_postdominators();
    print_ipostdominators();
  }
  printf("GPGPU-Sim PTX: pre-decoding instructions for \'%s\'...\n",
         m_name.c_str());
  //对m_instr_mem中的每一条指令进行预解码。
  for (unsigned ii = 0; ii < m_n;
       ii += m_instr_mem[ii]->inst_size()) {  // handle branch instructions
    ptx_instruction *pI = m_instr_mem[ii];
    pI->pre_decode();
  }
  printf("GPGPU-Sim PTX: ... done pre-decoding instructions for \'%s\'.\n",
         m_name.c_str());
  fflush(stdout);
  m_assembled = true;
}

void intersect(std::set<int> &A, const std::set<int> &B) {
  // return intersection of A and B in A
  for (std::set<int>::iterator a = A.begin(); a != A.end();) {
    std::set<int>::iterator a_next = a;
    a_next++;
    if (B.find(*a) == B.end()) {
      A.erase(*a);
      a = a_next;
    } else
      a++;
  }
}

bool is_equal(const std::set<int> &A, const std::set<int> &B) {
  if (A.size() != B.size()) return false;
  for (std::set<int>::iterator b = B.begin(); b != B.end(); b++)
    if (A.find(*b) == A.end()) return false;
  return true;
}

void print_set(const std::set<int> &A) {
  std::set<int>::iterator a;
  for (a = A.begin(); a != A.end(); a++) {
    printf("%d ", (*a));
  }
  printf("\n");
}

/*
必经结点（dominators）：如果从entry结点到结点i的每一条可能的执行路径都包含d，则结点d是结点i的必经结
                       点，记为d dom i。
直接必经结点（immediate dominators）：对于a≠b，当且仅当a dom b且不存在一个c≠a且c≠b的结点c，使得a 
                                     dom c且c dom b，则称a是b的直接必经结点，记为a idom b。
后必经结点（post-dominator）：从结点i到exit结点的每一条可能的执行路径都包含p，则结点p是结点i的后必经
                             结点，记为p pdom i。
寻找一个函数的完整PTX指令中，每一个（基本块）结点的必经结点（dominators）。例如，下述基本块之间的连接
图：
           entry
            \|/
   <———Yes—— B1 ——No———>
 \|/                   \|/
  B2                    B3
  |                    \|/
  |            <——No——— B4 <—
  |            |       \|/   |
  |            |       Yes   |
  |           \|/      \|/   |
  |            B5       B6 ——
 \|/__________\|/
       \|/
       exit
它的每一个基本块的必经结点集合为：
    i     |    Domin(i)
    entry |    {entry}
    B1    |    {entry,B1}
    B2    |    {entry,B1,B2}
    B3    |    {entry,B1,B3}
    B4    |    {entry,B1,B3,B4}
    B5    |    {entry,B1,B3,B4,B5}
    B6    |    {entry,B1,B3,B4,B6}
    exit  |    {entry,B1,exit}
*/
void function_info::find_dominators() {
  // find dominators using algorithm of Muchnick's Adv. Compiler Design &
  // Implemmntation Fig 7.14
  //使用了《高级编译器设计与实现(Steven.S.Muchnick著)》的图7.14中的算法。
  printf("GPGPU-Sim PTX: Finding dominators for \'%s\'...\n", m_name.c_str());
  fflush(stdout);
  assert(m_basic_blocks.size() >= 2);  // must have a distinquished entry block
  //基本块的迭代器。
  std::vector<basic_block_t *>::iterator bb_itr = m_basic_blocks.begin();
  //首先，bb_itr指向的是函数的入口基本块，这个入口基本块的必经结点集合中只有其自己。
  (*bb_itr)->dominator_ids.insert(
      (*bb_itr)->bb_id);  // the only dominator of the entry block is the entry
  // copy all basic blocks to all dominator lists EXCEPT for the entry block
  //依据图7.14中算法，初始化除入口基本块的其余所有基本块的必经结点集合，将其初始化为所有的基本块。即，
  //例如一个程序流图中有entry基本块、exit基本块、以及A/B基本块，初始化exit基本块、A基本块、B基本块的
  //的必经结点集合为{entry基本块、exit基本块、A基本块、B基本块}。
  for (++bb_itr; bb_itr != m_basic_blocks.end(); bb_itr++) {
    //将所有基本块都加入到除entry基本块外所有基本块的必经结点集合中。
    for (unsigned i = 0; i < m_basic_blocks.size(); i++)
      (*bb_itr)->dominator_ids.insert(i);
  }
  //以下是图7.14中算法的主体，书在腾讯文档GPGPU-Sim文档后面。下述各参数分别对应算法里的内容如下：
  //    [HERE]                            | [BOOK]
  //    change                            | change
  //    m_basic_blocks[h]                 | 结点n
  //    h                                 | 结点n的id
  //    std::set<int> T                   | T
  //    std::set<int>::iterator s         | p的id
  //    m_basic_blocks[*s]->dominator_ids | Domin(p)
  bool change = true;
  while (change) {
    change = false;
    for (int h = 1 /*skip entry*/; h < m_basic_blocks.size(); ++h) {
      assert(m_basic_blocks[h]->bb_id == (unsigned)h);
      std::set<int> T;
      for (unsigned i = 0; i < m_basic_blocks.size(); i++) T.insert(i);
      for (std::set<int>::iterator s =
               m_basic_blocks[h]->predecessor_ids.begin();
           s != m_basic_blocks[h]->predecessor_ids.end(); s++)
        intersect(T, m_basic_blocks[*s]->dominator_ids);
      T.insert(h);
      if (!is_equal(T, m_basic_blocks[h]->dominator_ids)) {
        change = true;
        m_basic_blocks[h]->dominator_ids = T;
      }
    }
  }
  // clean the basic block of dominators of it has no predecessors -- except for
  // entry block
  //上面的代码对算法中有改动，这里去除非Pred(n)的结点的必经结点。
  bb_itr = m_basic_blocks.begin();
  for (++bb_itr; bb_itr != m_basic_blocks.end(); bb_itr++) {
    if ((*bb_itr)->predecessor_ids.empty()) (*bb_itr)->dominator_ids.clear();
  }
}

/*
必经结点（dominators）：如果从entry结点到结点i的每一条可能的执行路径都包含d，则结点d是结点i的必经结
                       点，记为d dom i。
直接必经结点（immediate dominators）：对于a≠b，当且仅当a dom b且不存在一个c≠a且c≠b的结点c，使得a 
                                     dom c且c dom b，则称a是b的直接必经结点，记为a idom b。
后必经结点（post-dominator）：从结点i到exit结点的每一条可能的执行路径都包含p，则结点p是结点i的后必经
                             结点，记为p pdom i。
寻找一个函数的完整PTX指令中，每一个（基本块）结点的后必经结点（post-dominators）。例如，下述基本块之
间的连接图：
           entry
            \|/
   <———Yes—— B1 ——No———>
 \|/                   \|/
  B2                    B3
  |                    \|/
  |            <——No——— B4 <—
  |            |       \|/   |
  |            |       Yes   |
  |           \|/      \|/   |
  |            B5       B6 ——
 \|/__________\|/
       \|/
       exit
它的每一个基本块的后必经结点集合为：
    i     |    Domin(i)
    entry |    {exit,B1,entry}
    B1    |    {exit,B1}
    B2    |    {exit,B2}
    B3    |    {exit,B5,B4,B3}
    B4    |    {exit,B5,B4}
    B5    |    {exit,B5}
    B6    |    {exit,B5,B4,B6}
    exit  |    {exit}
*/
void function_info::find_postdominators() {
  // find postdominators using algorithm of Muchnick's Adv. Compiler Design &
  // Implemmntation Fig 7.14
  printf("GPGPU-Sim PTX: Finding postdominators for \'%s\'...\n",
         m_name.c_str());
  fflush(stdout);
  assert(m_basic_blocks.size() >= 2);  // must have a distinquished exit block
  std::vector<basic_block_t *>::reverse_iterator bb_itr =
      m_basic_blocks.rbegin();
  (*bb_itr)->postdominator_ids.insert(
      (*bb_itr)
          ->bb_id);  // the only postdominator of the exit block is the exit
  for (++bb_itr; bb_itr != m_basic_blocks.rend();
       bb_itr++) {  // copy all basic blocks to all postdominator lists EXCEPT
                    // for the exit block
    for (unsigned i = 0; i < m_basic_blocks.size(); i++)
      (*bb_itr)->postdominator_ids.insert(i);
  }
  bool change = true;
  while (change) {
    change = false;
    for (int h = m_basic_blocks.size() - 2 /*skip exit*/; h >= 0; --h) {
      assert(m_basic_blocks[h]->bb_id == (unsigned)h);
      std::set<int> T;
      for (unsigned i = 0; i < m_basic_blocks.size(); i++) T.insert(i);
      for (std::set<int>::iterator s = m_basic_blocks[h]->successor_ids.begin();
           s != m_basic_blocks[h]->successor_ids.end(); s++)
        intersect(T, m_basic_blocks[*s]->postdominator_ids);
      T.insert(h);
      if (!is_equal(T, m_basic_blocks[h]->postdominator_ids)) {
        change = true;
        m_basic_blocks[h]->postdominator_ids = T;
      }
    }
  }
}

/*
必经结点（dominators）：如果从entry结点到结点i的每一条可能的执行路径都包含d，则结点d是结点i的必经结
                       点，记为d dom i。
直接必经结点（immediate dominators）：对于a≠b，当且仅当a dom b且不存在一个c≠a且c≠b的结点c，使得a 
                                     dom c且c dom b，则称a是b的直接必经结点，记为a idom b。
后必经结点（post-dominator）：从结点i到exit结点的每一条可能的执行路径都包含p，则结点p是结点i的后必经
                             结点，记为p pdom i。
*/
void function_info::find_ipostdominators() {
  // find immediate postdominator blocks, using algorithm of
  // Muchnick's Adv. Compiler Design & Implemmntation Fig 7.15
  printf("GPGPU-Sim PTX: Finding immediate postdominators for \'%s\'...\n",
         m_name.c_str());
  fflush(stdout);
  assert(m_basic_blocks.size() >= 2);  // must have a distinquished exit block
  for (unsigned i = 0; i < m_basic_blocks.size();
       i++) {  // initialize Tmp(n) to all pdoms of n except for n
    m_basic_blocks[i]->Tmp_ids = m_basic_blocks[i]->postdominator_ids;
    assert(m_basic_blocks[i]->bb_id == i);
    m_basic_blocks[i]->Tmp_ids.erase(i);
  }
  for (int n = m_basic_blocks.size() - 2; n >= 0; --n) {
    // point iterator to basic block before the exit
    for (std::set<int>::iterator s = m_basic_blocks[n]->Tmp_ids.begin();
         s != m_basic_blocks[n]->Tmp_ids.end(); s++) {
      int bb_s = *s;
      for (std::set<int>::iterator t = m_basic_blocks[n]->Tmp_ids.begin();
           t != m_basic_blocks[n]->Tmp_ids.end();) {
        std::set<int>::iterator t_next = t;
        t_next++;  // might erase thing pointed to be t, invalidating iterator t
        if (*s == *t) {
          t = t_next;
          continue;
        }
        int bb_t = *t;
        if (m_basic_blocks[bb_s]->postdominator_ids.find(bb_t) !=
            m_basic_blocks[bb_s]->postdominator_ids.end())
          m_basic_blocks[n]->Tmp_ids.erase(bb_t);
        t = t_next;
      }
    }
  }
  unsigned num_ipdoms = 0;
  for (int n = m_basic_blocks.size() - 1; n >= 0; --n) {
    assert(m_basic_blocks[n]->Tmp_ids.size() <= 1);
    // if the above assert fails we have an error in either postdominator
    // computation, the flow graph does not have a unique exit, or some other
    // error
    if (!m_basic_blocks[n]->Tmp_ids.empty()) {
      m_basic_blocks[n]->immediatepostdominator_id =
          *m_basic_blocks[n]->Tmp_ids.begin();
      num_ipdoms++;
    }
  }
  assert(num_ipdoms == m_basic_blocks.size() - 1);
  // the exit node does not have an immediate post dominator, but everyone else
  // should
}

/*
必经结点（dominators）：如果从entry结点到结点i的每一条可能的执行路径都包含d，则结点d是结点i的必经结
                       点，记为d dom i。
直接必经结点（immediate dominators）：对于a≠b，当且仅当a dom b且不存在一个c≠a且c≠b的结点c，使得a 
                                     dom c且c dom b，则称a是b的直接必经结点，记为a idom b。
后必经结点（post-dominator）：从结点i到exit结点的每一条可能的执行路径都包含p，则结点p是结点i的后必经
                             结点，记为p pdom i。
寻找一个函数的完整PTX指令中，每一个（基本块）结点的直接必经结点（immediate dominators）。例如，下述
基本块之间的连接图：
           entry
            \|/
   <———Yes—— B1 ——No———>
 \|/                   \|/
  B2                    B3
  |                    \|/
  |            <——No——— B4 <—
  |            |       \|/   |
  |            |       Yes   |
  |           \|/      \|/   |
  |            B5       B6 ——
 \|/__________\|/
       \|/
       exit
它的每一个基本块的直接必经结点集合为：
    i     |    Domin(i)
    entry |    NULL
    B1    |    {entry}
    B2    |    {B1}
    B3    |    {B1}
    B4    |    {B3}
    B5    |    {B4}
    B6    |    {B4}
    exit  |    {B1}
*/
void function_info::find_idominators() {
  // find immediate dominator blocks, using algorithm of
  // Muchnick's Adv. Compiler Design & Implemmntation Fig 7.15
  printf("GPGPU-Sim PTX: Finding immediate dominators for \'%s\'...\n",
         m_name.c_str());
  fflush(stdout);
  assert(m_basic_blocks.size() >= 2);  // must have a distinquished entry block
  for (unsigned i = 0; i < m_basic_blocks.size();
       i++) {  // initialize Tmp(n) to all doms of n except for n
    m_basic_blocks[i]->Tmp_ids = m_basic_blocks[i]->dominator_ids;
    assert(m_basic_blocks[i]->bb_id == i);
    m_basic_blocks[i]->Tmp_ids.erase(i);
  }
  for (int n = 0; n < m_basic_blocks.size(); ++n) {
    // point iterator to basic block before the exit
    for (std::set<int>::iterator s = m_basic_blocks[n]->Tmp_ids.begin();
         s != m_basic_blocks[n]->Tmp_ids.end(); s++) {
      int bb_s = *s;
      for (std::set<int>::iterator t = m_basic_blocks[n]->Tmp_ids.begin();
           t != m_basic_blocks[n]->Tmp_ids.end();) {
        std::set<int>::iterator t_next = t;
        t_next++;  // might erase thing pointed to be t, invalidating iterator t
        if (*s == *t) {
          t = t_next;
          continue;
        }
        int bb_t = *t;
        if (m_basic_blocks[bb_s]->dominator_ids.find(bb_t) !=
            m_basic_blocks[bb_s]->dominator_ids.end())
          m_basic_blocks[n]->Tmp_ids.erase(bb_t);
        t = t_next;
      }
    }
  }
  unsigned num_idoms = 0;
  unsigned num_nopred = 0;
  for (int n = 0; n < m_basic_blocks.size(); ++n) {
    // assert( m_basic_blocks[n]->Tmp_ids.size() <= 1 );
    // if the above assert fails we have an error in either dominator
    // computation, the flow graph does not have a unique entry, or some other
    // error
    if (!m_basic_blocks[n]->Tmp_ids.empty()) {
      m_basic_blocks[n]->immediatedominator_id =
          *m_basic_blocks[n]->Tmp_ids.begin();
      num_idoms++;
    } else if (m_basic_blocks[n]->predecessor_ids.empty()) {
      num_nopred += 1;
    }
  }
  assert(num_idoms == m_basic_blocks.size() - num_nopred);
  // the entry node does not have an immediate dominator, but everyone else
  // should
}

void function_info::print_dominators() {
  printf("Printing dominators for function \'%s\':\n", m_name.c_str());
  std::vector<int>::iterator bb_itr;
  for (unsigned i = 0; i < m_basic_blocks.size(); i++) {
    printf("ID: %d\t:", i);
    for (std::set<int>::iterator j = m_basic_blocks[i]->dominator_ids.begin();
         j != m_basic_blocks[i]->dominator_ids.end(); j++)
      printf(" %d", *j);
    printf("\n");
  }
}

void function_info::print_postdominators() {
  printf("Printing postdominators for function \'%s\':\n", m_name.c_str());
  std::vector<int>::iterator bb_itr;
  for (unsigned i = 0; i < m_basic_blocks.size(); i++) {
    printf("ID: %d\t:", i);
    for (std::set<int>::iterator j =
             m_basic_blocks[i]->postdominator_ids.begin();
         j != m_basic_blocks[i]->postdominator_ids.end(); j++)
      printf(" %d", *j);
    printf("\n");
  }
}

void function_info::print_ipostdominators() {
  printf("Printing immediate postdominators for function \'%s\':\n",
         m_name.c_str());
  std::vector<int>::iterator bb_itr;
  for (unsigned i = 0; i < m_basic_blocks.size(); i++) {
    printf("ID: %d\t:", i);
    printf("%d\n", m_basic_blocks[i]->immediatepostdominator_id);
  }
}

void function_info::print_idominators() {
  printf("Printing immediate dominators for function \'%s\':\n",
         m_name.c_str());
  std::vector<int>::iterator bb_itr;
  for (unsigned i = 0; i < m_basic_blocks.size(); i++) {
    printf("ID: %d\t:", i);
    printf("%d\n", m_basic_blocks[i]->immediatedominator_id);
  }
}

unsigned function_info::get_num_reconvergence_pairs() {
  if (!num_reconvergence_pairs) {
    if (m_basic_blocks.size() == 0) return 0;
    for (unsigned i = 0; i < (m_basic_blocks.size() - 1);
         i++) {  // last basic block containing exit obviously won't have a pair
      if (m_basic_blocks[i]->ptx_end->get_opcode() == BRA_OP) {
        num_reconvergence_pairs++;
      }
    }
  }
  return num_reconvergence_pairs;
}

void function_info::get_reconvergence_pairs(gpgpu_recon_t *recon_points) {
  unsigned idx = 0;  // array index
  if (m_basic_blocks.size() == 0) return;
  for (unsigned i = 0; i < (m_basic_blocks.size() - 1);
       i++) {  // last basic block containing exit obviously won't have a pair
#ifdef DEBUG_GET_RECONVERG_PAIRS
    printf("i=%d\n", i);
    fflush(stdout);
#endif
    if (m_basic_blocks[i]->ptx_end->get_opcode() == BRA_OP) {
#ifdef DEBUG_GET_RECONVERG_PAIRS
      printf("\tbranch!\n");
      printf("\tbb_id=%d; ipdom=%d\n", m_basic_blocks[i]->bb_id,
             m_basic_blocks[i]->immediatepostdominator_id);
      printf("\tm_instr_mem index=%d\n",
             m_basic_blocks[i]->ptx_end->get_m_instr_mem_index());
      fflush(stdout);
#endif
      recon_points[idx].source_pc = m_basic_blocks[i]->ptx_end->get_PC();
      recon_points[idx].source_inst = m_basic_blocks[i]->ptx_end;
#ifdef DEBUG_GET_RECONVERG_PAIRS
      printf("\trecon_points[idx].source_pc=%d\n", recon_points[idx].source_pc);
#endif
      if (m_basic_blocks[m_basic_blocks[i]->immediatepostdominator_id]
              ->ptx_begin) {
        recon_points[idx].target_pc =
            m_basic_blocks[m_basic_blocks[i]->immediatepostdominator_id]
                ->ptx_begin->get_PC();
        recon_points[idx].target_inst =
            m_basic_blocks[m_basic_blocks[i]->immediatepostdominator_id]
                ->ptx_begin;
      } else {
        // reconverge after function return
        recon_points[idx].target_pc = -2;
        recon_points[idx].target_inst = NULL;
      }
#ifdef DEBUG_GET_RECONVERG_PAIRS
      m_basic_blocks[m_basic_blocks[i]->immediatepostdominator_id]
          ->ptx_begin->print_insn();
      printf("\trecon_points[idx].target_pc=%d\n", recon_points[idx].target_pc);
      fflush(stdout);
#endif
      idx++;
    }
  }
}

// interface with graphviz (print the graph in DOT language) for plotting
void function_info::print_basic_block_dot() {
  printf("Basic Block in DOT\n");
  printf("digraph %s {\n", m_name.c_str());
  std::vector<basic_block_t *>::iterator bb_itr;
  for (bb_itr = m_basic_blocks.begin(); bb_itr != m_basic_blocks.end();
       bb_itr++) {
    printf("\t");
    std::set<int>::iterator s;
    for (s = (*bb_itr)->successor_ids.begin();
         s != (*bb_itr)->successor_ids.end(); s++) {
      unsigned succ_bb = *s;
      printf("%d -> %d; ", (*bb_itr)->bb_id, succ_bb);
    }
    printf("\n");
  }
  printf("}\n");
}

unsigned ptx_kernel_shmem_size(void *kernel_impl) {
  function_info *f = (function_info *)kernel_impl;
  const struct gpgpu_ptx_sim_info *kernel_info = f->get_kernel_info();
  return kernel_info->smem;
}

unsigned ptx_kernel_nregs(void *kernel_impl) {
  function_info *f = (function_info *)kernel_impl;
  const struct gpgpu_ptx_sim_info *kernel_info = f->get_kernel_info();
  return kernel_info->regs;
}

unsigned type_info_key::type_decode(size_t &size, int &basic_type) const {
  int type = scalar_type();
  return type_decode(type, size, basic_type);
}

unsigned type_info_key::type_decode(int type, size_t &size, int &basic_type) {
  switch (type) {
    case S8_TYPE:
      size = 8;
      basic_type = 1;
      return 0;
    case S16_TYPE:
      size = 16;
      basic_type = 1;
      return 1;
    case S32_TYPE:
      size = 32;
      basic_type = 1;
      return 2;
    case S64_TYPE:
      size = 64;
      basic_type = 1;
      return 3;
    case U8_TYPE:
      size = 8;
      basic_type = 0;
      return 4;
    case U16_TYPE:
      size = 16;
      basic_type = 0;
      return 5;
    case U32_TYPE:
      size = 32;
      basic_type = 0;
      return 6;
    case U64_TYPE:
      size = 64;
      basic_type = 0;
      return 7;
    case F16_TYPE:
      size = 16;
      basic_type = -1;
      return 8;
    case F32_TYPE:
      size = 32;
      basic_type = -1;
      return 9;
    case F64_TYPE:
      size = 64;
      basic_type = -1;
      return 10;
    case FF64_TYPE:
      size = 64;
      basic_type = -1;
      return 10;
    case PRED_TYPE:
      size = 1;
      basic_type = 2;
      return 11;
    case B8_TYPE:
      size = 8;
      basic_type = 0;
      return 12;
    case B16_TYPE:
      size = 16;
      basic_type = 0;
      return 13;
    case B32_TYPE:
      size = 32;
      basic_type = 0;
      return 14;
    case B64_TYPE:
      size = 64;
      basic_type = 0;
      return 15;
    case BB64_TYPE:
      size = 64;
      basic_type = 0;
      return 15;
    case BB128_TYPE:
      size = 128;
      basic_type = 0;
      return 16;
    case TEXREF_TYPE:
    case SAMPLERREF_TYPE:
    case SURFREF_TYPE:
      size = 32;
      basic_type = 3;
      return 16;
    default:
      printf("ERROR ** type_decode() does not know about \"%s\"\n",
             decode_token(type));
      assert(0);
      return 0xDEADBEEF;
  }
}

arg_buffer_t copy_arg_to_buffer(ptx_thread_info *thread,
                                operand_info actual_param_op,
                                const symbol *formal_param) {
  if (actual_param_op.is_reg()) {
    ptx_reg_t value = thread->get_reg(actual_param_op.get_symbol());
    return arg_buffer_t(formal_param, actual_param_op, value);
  } else if (actual_param_op.is_param_local()) {
    unsigned size = formal_param->get_size_in_bytes();
    addr_t frame_offset = actual_param_op.get_symbol()->get_address();
    addr_t from_addr = thread->get_local_mem_stack_pointer() + frame_offset;
    char buffer[1024];
    assert(size < 1024);
    thread->m_local_mem->read(from_addr, size, buffer);
    return arg_buffer_t(formal_param, actual_param_op, buffer, size);
  } else {
    printf(
        "GPGPU-Sim PTX: ERROR ** need to add support for this operand type in "
        "call/return\n");
    abort();
  }
}

void copy_args_into_buffer_list(const ptx_instruction *pI,
                                ptx_thread_info *thread,
                                const function_info *target_func,
                                arg_buffer_list_t &arg_values) {
  unsigned n_return = target_func->has_return();
  unsigned n_args = target_func->num_args();
  for (unsigned arg = 0; arg < n_args; arg++) {
    const operand_info &actual_param_op =
        pI->operand_lookup(n_return + 1 + arg);
    const symbol *formal_param = target_func->get_arg(arg);
    arg_values.push_back(
        copy_arg_to_buffer(thread, actual_param_op, formal_param));
  }
}

void copy_buffer_to_frame(ptx_thread_info *thread, const arg_buffer_t &a) {
  if (a.is_reg()) {
    ptx_reg_t value = a.get_reg();
    operand_info dst_reg =
        operand_info(a.get_dst(), thread->get_gpu()->gpgpu_ctx);
    thread->set_reg(dst_reg.get_symbol(), value);
  } else {
    const void *buffer = a.get_param_buffer();
    size_t size = a.get_param_buffer_size();
    const symbol *dst = a.get_dst();
    addr_t frame_offset = dst->get_address();
    addr_t to_addr = thread->get_local_mem_stack_pointer() + frame_offset;
    thread->m_local_mem->write(to_addr, size, buffer, NULL, NULL);
  }
}

void copy_buffer_list_into_frame(ptx_thread_info *thread,
                                 arg_buffer_list_t &arg_values) {
  arg_buffer_list_t::iterator a;
  for (a = arg_values.begin(); a != arg_values.end(); a++) {
    copy_buffer_to_frame(thread, *a);
  }
}

static std::list<operand_info> check_operands(
    int opcode, const std::list<int> &scalar_type,
    const std::list<operand_info> &operands, gpgpu_context *ctx) {
  static int g_warn_literal_operands_two_type_inst;
  if ((opcode == CVT_OP) || (opcode == SET_OP) || (opcode == SLCT_OP) ||
      (opcode == TEX_OP) || (opcode == MMA_OP) || (opcode == DP4A_OP) || 
      (opcode == CIMMA_OP) || //yangjianchao16
      (opcode == VMIN_OP) || (opcode == VMAX_OP) ) {
    // just make sure these do not have have const operands...
    if (!g_warn_literal_operands_two_type_inst) {
      std::list<operand_info>::const_iterator o;
      for (o = operands.begin(); o != operands.end(); o++) {
        const operand_info &op = *o;
        if (op.is_literal()) {
          printf(
              "GPGPU-Sim PTX: PTX uses two scalar type intruction with literal "
              "operand.\n");
          g_warn_literal_operands_two_type_inst = 1;
        }
      }
    }
  } else {
    assert(scalar_type.size() < 2);
    if (scalar_type.size() == 1) {
      std::list<operand_info> result;
      int inst_type = scalar_type.front();
      std::list<operand_info>::const_iterator o;
      for (o = operands.begin(); o != operands.end(); o++) {
        const operand_info &op = *o;
        if (op.is_literal()) {
          if ((op.get_type() == double_op_t) && (inst_type == F32_TYPE)) {
            ptx_reg_t v = op.get_literal_value();
            float u = (float)v.f64;
            operand_info n(u, ctx);
            result.push_back(n);
          } else {
            result.push_back(op);
          }
        } else {
          result.push_back(op);
        }
      }
      return result;
    }
  }
  return operands;
}

ptx_instruction::ptx_instruction(
    int opcode, const symbol *pred, int neg_pred, int pred_mod, symbol *label,
    const std::list<operand_info> &operands, const operand_info &return_var,
    const std::list<int> &options, const std::list<int> &wmma_options,
    const std::list<int> &cimma_options, //yangjianchao16
    const std::list<int> &scalar_type, memory_space_t space_spec,
    const char *file, unsigned line, const char *source,
    const core_config *config, gpgpu_context *ctx)
    : warp_inst_t(config), m_return_var(ctx) {
  gpgpu_ctx = ctx;
  m_uid = ++(ctx->g_num_ptx_inst_uid);
  m_PC = 0;
  m_opcode = opcode;
  m_pred = pred;
  m_neg_pred = neg_pred;
  m_pred_mod = pred_mod;
  m_label = label;
  const std::list<operand_info> checked_operands =
      check_operands(opcode, scalar_type, operands, ctx);
  m_operands.insert(m_operands.begin(), checked_operands.begin(),
                    checked_operands.end());
  m_return_var = return_var;
  m_options = options;
  m_wmma_options = wmma_options;
  m_cimma_options = cimma_options; //yangjianchao16
  m_wide = false;
  m_hi = false;
  m_lo = false;
  m_uni = false;
  m_exit = false;
  m_abs = false;
  m_neg = false;
  m_to_option = false;
  m_cache_option = 0;
  m_rounding_mode = RN_OPTION;
  m_compare_op = -1;
  m_saturation_mode = 0;
  m_geom_spec = 0;
  m_vector_spec = 0;
  m_atomic_spec = 0;
  m_membar_level = 0;
  m_inst_size = 8;  // bytes
  int rr = 0;
  std::list<int>::const_iterator i;
  unsigned n = 1;
  for (i = wmma_options.begin(); i != wmma_options.end(); i++, n++) {
    int last_ptx_inst_option = *i;
    switch (last_ptx_inst_option) {
      case SYNC_OPTION:
      case LOAD_A:
      case LOAD_B:
      case LOAD_C:
      case STORE_D:
      case MMA:
        m_wmma_type = last_ptx_inst_option;
        break;
      case ROW:
      case COL:
        m_wmma_layout[rr++] = last_ptx_inst_option;
        break;
      case M16N16K16:
      case M32N8K16:
      case M8N32K16:
        break;
      default:
        assert(0);
        break;
    }
  }

  // cimma.shmma.synchro.rowmajor.colmajor.m128n128k8.f32.f32 %rd39 %rd37 %rd38; //yangjianchao16
  n = 1;                                                                         //yangjianchao16
  rr = 0;                                                                        //yangjianchao16
  for (i = cimma_options.begin(); i != cimma_options.end(); i++, n++) {          //yangjianchao16
    int last_ptx_inst_option = *i;                                               //yangjianchao16
    switch (last_ptx_inst_option) {                                              //yangjianchao16
      case SHMMA:                                                                //yangjianchao16
        m_cimma_type = last_ptx_inst_option;                                      //yangjianchao16
        break;                                                                   //yangjianchao16
      case ROWMAJOR:                                                             //yangjianchao16
      case COLMAJOR:                                                             //yangjianchao16
        m_cimma_layout[rr++] = last_ptx_inst_option;                              //yangjianchao16
        break;                                                                   //yangjianchao16
      case M128N128K8:                                                           //yangjianchao16
        break;                                                                   //yangjianchao16
      default:                                                                   //yangjianchao16
        assert(0);                                                               //yangjianchao16
        break;                                                                   //yangjianchao16
    }                                                                            //yangjianchao16
  }                                                                              //yangjianchao16

  rr = 0;
  n = 1;
  for (i = options.begin(); i != options.end(); i++, n++) {
    int last_ptx_inst_option = *i;
    switch (last_ptx_inst_option) {
      case SYNC_OPTION:
      case ARRIVE_OPTION:
      case RED_OPTION:
        m_barrier_op = last_ptx_inst_option;
        break;
      case EQU_OPTION:
      case NEU_OPTION:
      case LTU_OPTION:
      case LEU_OPTION:
      case GTU_OPTION:
      case GEU_OPTION:
      case EQ_OPTION:
      case NE_OPTION:
      case LT_OPTION:
      case LE_OPTION:
      case GT_OPTION:
      case GE_OPTION:
      case LS_OPTION:
      case HS_OPTION:
        m_compare_op = last_ptx_inst_option;
        break;
      case NUM_OPTION:
      case NAN_OPTION:
        m_compare_op = last_ptx_inst_option;
        // assert(0); // finish this
        break;
      case SAT_OPTION:
        m_saturation_mode = 1;
        break;
      case RNI_OPTION:
      case RZI_OPTION:
      case RMI_OPTION:
      case RPI_OPTION:
      case RN_OPTION:
      case RZ_OPTION:
      case RM_OPTION:
      case RP_OPTION:
        m_rounding_mode = last_ptx_inst_option;
        break;
      case HI_OPTION:
        m_compare_op = last_ptx_inst_option;
        m_hi = true;
        assert(!m_lo);
        assert(!m_wide);
        break;
      case LO_OPTION:
        m_compare_op = last_ptx_inst_option;
        m_lo = true;
        assert(!m_hi);
        assert(!m_wide);
        break;
      case WIDE_OPTION:
        m_wide = true;
        assert(!m_lo);
        assert(!m_hi);
        break;
      case UNI_OPTION:
        m_uni = true;  // don't care... < now we DO care when constructing
                       // flowgraph>
        break;
      case GEOM_MODIFIER_1D:
      case GEOM_MODIFIER_2D:
      case GEOM_MODIFIER_3D:
        m_geom_spec = last_ptx_inst_option;
        break;
      case V2_TYPE:
      case V3_TYPE:
      case V4_TYPE:
        m_vector_spec = last_ptx_inst_option;
        break;
      case ATOMIC_AND:
      case ATOMIC_OR:
      case ATOMIC_XOR:
      case ATOMIC_CAS:
      case ATOMIC_EXCH:
      case ATOMIC_ADD:
      case ATOMIC_INC:
      case ATOMIC_DEC:
      case ATOMIC_MIN:
      case ATOMIC_MAX:
        m_atomic_spec = last_ptx_inst_option;
        break;
      case APPROX_OPTION:
        break;
      case FULL_OPTION:
        break;
      case ANY_OPTION:
        m_vote_mode = vote_any;
        break;
      case ALL_OPTION:
        m_vote_mode = vote_all;
        break;
      case BALLOT_OPTION:
        m_vote_mode = vote_ballot;
        break;
      case GLOBAL_OPTION:
        m_membar_level = GLOBAL_OPTION;
        break;
      case CTA_OPTION:
        m_membar_level = CTA_OPTION;
        break;
      case SYS_OPTION:
        m_membar_level = SYS_OPTION;
        break;
      case FTZ_OPTION:
        break;
      case EXIT_OPTION:
        m_exit = true;
        break;
      case ABS_OPTION:
        m_abs = true;
        break;
      case NEG_OPTION:
        m_neg = true;
        break;
      case TO_OPTION:
        m_to_option = true;
        break;
      case CA_OPTION:
      case CG_OPTION:
      case CS_OPTION:
      case LU_OPTION:
      case CV_OPTION:
      case WB_OPTION: 
      case WT_OPTION:
        m_cache_option = last_ptx_inst_option;
        break;
      case HALF_OPTION:
        m_inst_size = 4;  // bytes
        break;
      case EXTP_OPTION:
        break;
      case NC_OPTION:
        m_cache_option = last_ptx_inst_option;
        break;
      case UP_OPTION:
      case DOWN_OPTION:
      case BFLY_OPTION:
      case IDX_OPTION:
        m_shfl_op = last_ptx_inst_option;
        break;
      case PRMT_F4E_MODE:
      case PRMT_B4E_MODE:
      case PRMT_RC8_MODE:
      case PRMT_ECL_MODE:
      case PRMT_ECR_MODE:
      case PRMT_RC16_MODE:
        m_prmt_op = last_ptx_inst_option;
        break;
      default:
        assert(0);
        break;
    }
  }
  m_scalar_type = scalar_type;
  m_space_spec = space_spec;
  if ((opcode == ST_OP || opcode == LD_OP || opcode == LDU_OP) &&
      (space_spec == undefined_space)) {
    m_space_spec = generic_space;
  }
  for (std::vector<operand_info>::const_iterator i = m_operands.begin();
       i != m_operands.end(); ++i) {
    const operand_info &op = *i;
    if (op.get_addr_space() != undefined_space)
      m_space_spec =
          op.get_addr_space();  // TODO: can have more than one memory space for
                                // ptxplus (g8x) inst
  }
  if (opcode == TEX_OP) m_space_spec = tex_space;

  m_source_file = file ? file : "<unknown>";
  m_source_line = line;
  m_source = source;
  // Trim tabs
  m_source.erase(std::remove(m_source.begin(), m_source.end(), '\t'),
                 m_source.end());

  if (opcode == CALL_OP) {
    const operand_info &target = func_addr();
    assert(target.is_function_address());
    const symbol *func_addr = target.get_symbol();
    const function_info *target_func = func_addr->get_pc();
    std::string fname = target_func->get_name();

    if (fname == "vprintf") {
      m_is_printf = true;
    }
    if (fname == "cudaStreamCreateWithFlags") m_is_cdp = 1;
    if (fname == "cudaGetParameterBufferV2") m_is_cdp = 2;
    if (fname == "cudaLaunchDeviceV2") m_is_cdp = 4;
  }
}

/*
调用ptx_instruction::print_insn(FILE *fp)，传入参数stdout，打印指令到屏幕。stdin,stdout,stderr就是
这个fp，不过它是随着计算机系统的开启默认打开的：其中0就是stdin，表示输入流，指从键盘输入；1代表stdout；
2代表stderr；1，2默认是显示器。
*/
void ptx_instruction::print_insn() const {
  print_insn(stdout);
  fflush(stdout);
}

/*
传入参数 FILE *fp，打印指令到该文件。
*/
void ptx_instruction::print_insn(FILE *fp) const {
  fprintf(fp, "%s\n", to_string().c_str());
}

/*
将该指令转换为一个字符串，并返回该字符串。C++ 库函数 int snprintf(char *str, size_t size, const char 
*format, ...) 设将可变参数(...)按照 format 格式化成字符串，并将字符串复制到 str 中，size 为要写入的字
符的最大数目，超过 size 会被截断。
*/
std::string ptx_instruction::to_string() const {
  char buf[STR_SIZE];
  unsigned used_bytes = 0;
  if (!is_label()) {
    used_bytes +=
        snprintf(buf + used_bytes, STR_SIZE - used_bytes, " PC=0x%03x ", m_PC);
        //printf("\n== In ptx_instruction::to_string() !is_label() ==> PC=0x%03x \n", m_PC);
  } else {
    used_bytes +=
        snprintf(buf + used_bytes, STR_SIZE - used_bytes, "                ");
        //printf("\n== In ptx_instruction::to_string() is_label() ==>  \n");
  }
  used_bytes +=
      snprintf(buf + used_bytes, STR_SIZE - used_bytes, "(%s:%d) %s",
               m_source_file.c_str(), m_source_line, m_source.c_str());
      //printf("\n== In ptx_instruction::to_string() all ==> (%s:%d) %s \n", 
      //       m_source_file.c_str(), m_source_line, m_source.c_str());
  return std::string(buf);
}

/*
获取谓词。将谓词作为一个 operand_info 对象操作数信息返回。
*/
operand_info ptx_instruction::get_pred() const {
  return operand_info(m_pred, gpgpu_ctx);
}

/*
function_info的构造函数。
*/
function_info::function_info(int entry_point, gpgpu_context *ctx) {
  gpgpu_ctx = ctx;
  m_uid = (gpgpu_ctx->function_info_sm_next_uid)++;
  m_entry_point = (entry_point == 1) ? true : false;
  m_extern = (entry_point == 2) ? true : false;
  num_reconvergence_pairs = 0;
  m_symtab = NULL;
  m_assembled = false;
  m_return_var_sym = NULL;
  m_kernel_info.cmem = 0;
  m_kernel_info.lmem = 0;
  m_kernel_info.regs = 0;
  m_kernel_info.smem = 0;
  m_local_mem_framesize = 0;
  m_args_aligned_size = -1;
  //初始化寻找后支配者完成状态为 False。
  pdom_done = false;  // initialize it to false
}

unsigned function_info::print_insn(unsigned pc, FILE *fp) const {
  unsigned inst_size = 1;  // return offset to next instruction or 1 if unknown
  unsigned index = pc - m_start_PC;
  char command[1024];
  char buffer[1024];
  memset(command, 0, 1024);
  memset(buffer, 0, 1024);
  snprintf(command, 1024, "c++filt -p %s", m_name.c_str());
  FILE *p = popen(command, "r");
  buffer[0] = 0;
  assert(fgets(buffer, 1023, p) != NULL);
  // Remove trailing "\n" in buffer
  char *c;
  if ((c = strchr(buffer, '\n')) != NULL) *c = '\0';
  fprintf(fp, "%s", buffer);
  if (index >= m_instr_mem_size) {
    fprintf(fp, "<past last instruction (max pc=%u)>",
            m_start_PC + m_instr_mem_size - 1);
  } else {
    if (m_instr_mem[index] != NULL) {
      m_instr_mem[index]->print_insn(fp);
      inst_size = m_instr_mem[index]->isize;
    } else
      fprintf(fp, "<no instruction at pc = %u>", pc);
  }
  pclose(p);
  return inst_size;
}

std::string function_info::get_insn_str(unsigned pc) const {
  unsigned index = pc - m_start_PC;
  if (index >= m_instr_mem_size) {
    char buff[STR_SIZE];
    buff[STR_SIZE - 1] = '\0';
    snprintf(buff, STR_SIZE, "<past last instruction (max pc=%u)>",
             m_start_PC + m_instr_mem_size - 1);
    return std::string(buff);
  } else {
    if (m_instr_mem[index] != NULL) {
      return m_instr_mem[index]->to_string();
    } else {
      char buff[STR_SIZE];
      buff[STR_SIZE - 1] = '\0';
      snprintf(buff, STR_SIZE, "<no instruction at pc = %u>", pc);
      return std::string(buff);
    }
  }
}

void gpgpu_ptx_assemble(std::string kname, void *kinfo) {
  function_info *func_info = (function_info *)kinfo;
  if ((function_info *)kinfo == NULL) {
    printf("GPGPU-Sim PTX: Warning - missing function definition \'%s\'\n",
           kname.c_str());
    return;
  }
  if (func_info->is_extern()) {
    printf(
        "GPGPU-Sim PTX: skipping assembly for extern declared function "
        "\'%s\'\n",
        func_info->get_name().c_str());
    return;
  }
  func_info->ptx_assemble();
}
